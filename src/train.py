import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import SEED, get_data_splits
from model import build_baseline_cnn, compile_baseline_cnn
from transfer_model import build_transfer_model, unfreeze_last_block

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
EPOCHS_BASELINE = 30
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
LR_BASELINE = 1e-3
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5


def set_seeds(seed=SEED):
    tf.keras.utils.set_random_seed(seed)


def get_callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / f"{name}.keras", monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR / f"{name}_history.csv"),
    ]


def plot_history(history, name, title):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    best_epoch = int(np.argmin(h["val_loss"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, h[metric], marker="o", markersize=3, label=f"Train {label.lower()}")
        ax.plot(epochs, h[f"val_{metric}"], marker="o", markersize=3, label=f"Val {label.lower()}")
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_curves.png", dpi=150)
    plt.close(fig)
    return best_epoch


def save_trainable_layers(model, base, name):
    trainable_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    lines = [f"{name}: {trainable_params:,} trainable parameters", "Trainable layers:"]
    for layer in model.layers:
        if layer is base:
            for sub in base.layers:
                if sub.trainable_weights:
                    lines.append(f"  {base.name}/{sub.name}")
        elif layer.trainable_weights:
            lines.append(f"  {layer.name}")
    text = "\n".join(lines)
    (OUTPUT_DIR / f"{name}_trainable_layers.txt").write_text(text + "\n")
    print("\n" + text + "\n")


def summarize(name, title, model, history, val_ds):
    best_epoch = plot_history(history, name, title)
    val_results = model.evaluate(val_ds, return_dict=True, verbose=0)
    summary = {
        "model": name,
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        **{f"val_{k}": round(float(v), 4) for k, v in val_results.items()},
    }
    (OUTPUT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{title}: best epoch {best_epoch}")
    print(json.dumps(summary, indent=2))
    return summary


def train_baseline():
    set_seeds()
    train_ds, val_ds, _, _ = get_data_splits(normalize=True)
    model = compile_baseline_cnn(build_baseline_cnn(), learning_rate=LR_BASELINE)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_BASELINE,
        callbacks=get_callbacks("baseline_cnn"),
    )
    model.load_weights(MODEL_DIR / "baseline_cnn.keras")
    return summarize("baseline_cnn", "Baseline CNN", model, history, val_ds)


def train_transfer():
    set_seeds()
    train_ds, val_ds, _, _ = get_data_splits(normalize=False)
    model, base = build_transfer_model()

    compile_baseline_cnn(model, learning_rate=LR_STAGE1)
    save_trainable_layers(model, base, "transfer_stage1")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1,
        callbacks=get_callbacks("transfer_stage1"),
    )
    model.load_weights(MODEL_DIR / "transfer_stage1.keras")
    stage1 = summarize(
        "transfer_stage1", "MobileNetV2 stage 1 (frozen backbone)", model, history1, val_ds
    )

    unfreeze_last_block(base)
    compile_baseline_cnn(model, learning_rate=LR_STAGE2)
    save_trainable_layers(model, base, "transfer_stage2")
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2,
        callbacks=get_callbacks("transfer_stage2"),
    )
    model.load_weights(MODEL_DIR / "transfer_stage2.keras")
    stage2 = summarize(
        "transfer_stage2", "MobileNetV2 stage 2 (last block unfrozen)", model, history2, val_ds
    )
    return stage1, stage2


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    if choice in ("baseline", "all"):
        train_baseline()
    if choice in ("transfer", "all"):
        train_transfer()