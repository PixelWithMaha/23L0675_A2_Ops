"""Tasks 3.3 / 3.4 / 3.5: VisionGuard training tracked with MLflow (DagsHub or local).

Manual logging (Keras callback):   python src/train_mlflow.py --run-name mnv2-manual
MLflow TensorFlow autologging:     python src/train_mlflow.py --mode autolog --run-name mnv2-autolog
One-change variations (3.5):       add e.g. --lr2 1e-4, --no-augment, --unfrozen-blocks 2, --image-size 160
"""
import argparse
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from dataset import SEED, SPLIT_DIR, get_data_splits, read_split
from model import build_baseline_cnn, compile_baseline_cnn
from transfer_model import build_transfer_model, unfreeze_last_block

load_dotenv()  # MLFLOW_TRACKING_URI / USERNAME / PASSWORD come from .env, never from code
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "visionguard"
CLASS_NAMES = ["AiArtData", "RealArt"]
THRESHOLD = 0.5
UNFREEZE_FROM = {1: "block_16_expand", 2: "block_15_expand", 3: "block_14_expand"}


# ---------------------------------------------------------------- helpers
class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual logging: send train and val metrics to MLflow after every epoch."""

    def __init__(self, step_offset=0):
        super().__init__()
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(
            {k: float(v) for k, v in (logs or {}).items()}, step=self.step_offset + epoch + 1
        )


def get_callbacks(checkpoint_path, patience):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 3), min_lr=1e-7
        ),
    ]


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def dvc_data_version(dvc_file="data.dvc"):
    try:
        return yaml.safe_load(Path(dvc_file).read_text())["outs"][0]["md5"]
    except Exception:
        return "unknown"


def save_package_list(path):
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    path.write_text("\n".join(packages) + "\n")


def plot_curves(histories, path):
    loss, val_loss, acc, val_acc, boundaries = [], [], [], [], []
    for h in histories:
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        boundaries.append(len(loss))
    epochs = range(1, len(loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, tr, va, name in [(axes[0], loss, val_loss, "Loss"), (axes[1], acc, val_acc, "Accuracy")]:
        ax.plot(epochs, tr, marker="o", markersize=3, label=f"Train {name.lower()}")
        ax.plot(epochs, va, marker="o", markersize=3, label=f"Val {name.lower()}")
        for b in boundaries[:-1]:
            ax.axvline(b + 0.5, color="grey", linestyle=":", label="Stage 2 starts")
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_and_plot(model, test_ds, out_dir):
    _, y = read_split("test")
    y = np.array(y)
    p = model.predict(test_ds, verbose=0).ravel()
    pred = (p >= THRESHOLD).astype(int)
    metrics = {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_auc": roc_auc_score(y, p),
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"Test confusion matrix (threshold {THRESHOLD})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['test_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    return metrics


def fit_stage(model, train_ds, val_ds, epochs, patience, ckpt, args, step_offset, stage_name):
    callbacks = get_callbacks(ckpt, patience)
    if args.mode == "manual":
        callbacks.append(MLflowEpochLogger(step_offset))
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    else:
        # Autolog logs each fit() into its own nested run (params cannot change inside one run)
        with mlflow.start_run(run_name=stage_name, nested=True):
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.load_weights(ckpt)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    mlflow.log_metrics({f"best_epoch_{stage_name}": best_epoch,
                        f"best_val_loss_{stage_name}": float(min(history.history["val_loss"]))})
    return history


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transfer", "baseline"], default="transfer")
    parser.add_argument("--mode", choices=["manual", "autolog"], default="manual")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--lr1", type=float, default=1e-3, help="baseline / stage 1 learning rate")
    parser.add_argument("--lr2", type=float, default=1e-5, help="stage 2 (fine-tuning) learning rate")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfrozen-blocks", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--epochs1", type=int, default=15)
    parser.add_argument("--epochs2", type=int, default=15)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    print(f"Tracking to: {TRACKING_URI}")

    size = (args.image_size, args.image_size)
    augment = not args.no_augment
    train_ds, val_ds, test_ds, _ = get_data_splits(
        normalize=(args.model == "baseline"), image_size=size, augment=augment
    )

    if args.mode == "autolog":
        mlflow.tensorflow.autolog(log_models=False)  # the final model is logged once below

    run_name = args.run_name or f"{args.model}-{args.mode}"
    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.set_tags({
            "git_commit": git_commit(),
            "dvc_data_version": dvc_data_version(),
            "seed": SEED,
            "logging_mode": args.mode,
            "model_type": args.model,
        })
        params = {
            "model": args.model, "image_size": args.image_size, "augmentation": augment,
            "batch_size": 32, "seed": SEED, "threshold": THRESHOLD,
            "lr_stage1": args.lr1, "epochs_stage1": args.epochs1,
        }
        if args.model == "transfer":
            params.update({
                "backbone": "MobileNetV2", "dropout": args.dropout, "lr_stage2": args.lr2,
                "unfrozen_blocks": args.unfrozen_blocks, "epochs_stage2": args.epochs2,
            })
        if args.mode == "manual":
            mlflow.log_params(params)
        else:
            mlflow.log_params({k: v for k, v in params.items()
                               if k in ("model", "image_size", "augmentation", "seed", "threshold",
                                        "backbone", "dropout", "unfrozen_blocks")})

        histories = []
        if args.model == "baseline":
            model = compile_baseline_cnn(build_baseline_cnn((*size, 3)), learning_rate=args.lr1)
            histories.append(fit_stage(model, train_ds, val_ds, args.epochs1, 10,
                                       tmp / "best.keras", args, 0, "stage1"))
        else:
            model, base = build_transfer_model((*size, 3), dropout=args.dropout)
            compile_baseline_cnn(model, learning_rate=args.lr1)
            h1 = fit_stage(model, train_ds, val_ds, args.epochs1, 5,
                           tmp / "stage1.keras", args, 0, "stage1")
            histories.append(h1)

            unfreeze_last_block(base, UNFREEZE_FROM[args.unfrozen_blocks])
            compile_baseline_cnn(model, learning_rate=args.lr2)
            histories.append(fit_stage(model, train_ds, val_ds, args.epochs2, 5,
                                       tmp / "stage2.keras", args, len(h1.history["loss"]), "stage2"))

        # Final test metrics (threshold 0.5, as decided in Task 2.5)
        test_metrics = evaluate_and_plot(model, test_ds, tmp)
        mlflow.log_metrics(test_metrics)

        # 3.4 artifacts: curves, confusion matrix, ROC, split lists, package list, model
        plot_curves(histories, tmp / "training_curves.png")
        save_package_list(tmp / "packages.txt")
        for f in ["training_curves.png", "confusion_matrix.png", "roc_curve.png"]:
            mlflow.log_artifact(str(tmp / f), artifact_path="plots")
        for split in ["train", "val", "test"]:
            mlflow.log_artifact(str(SPLIT_DIR / f"{split}.csv"), artifact_path="splits")
        mlflow.log_artifact(str(tmp / "packages.txt"), artifact_path="environment")
        mlflow.keras.log_model(model, name="model")

        print(f"\nRun '{run_name}' ({run.info.run_id}) finished")
        for k, v in test_metrics.items():
            print(f"  {k:15s} {v:.4f}")


if __name__ == "__main__":
    sys.exit(main())
