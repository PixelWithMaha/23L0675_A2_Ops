import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset import build_pipeline, read_split

MODEL_DIR = Path("models")
OUT_DIR = Path("outputs/evaluation")
CLASS_NAMES = ["AiArtData", "RealArt"]  

MODELS = {
    "baseline_cnn": {"file": "baseline_cnn.keras", "normalize": True, "title": "Baseline CNN"},
    "transfer_stage2": {
        "file": "transfer_stage2.keras", "normalize": False, "title": "MobileNetV2 (fine-tuned)"
    },
}

COST_REAL_AS_FAKE = 2.0
COST_FAKE_AS_REAL = 1.0
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def predict(model, split, normalize):
    paths, labels = read_split(split)
    ds = build_pipeline(paths, labels, is_training=False, normalize=normalize)
    probs = model.predict(ds, verbose=0).ravel()
    return np.array(paths), np.array(labels), probs


def choose_threshold(y, p):
    rows = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(int)
        real_as_fake = int(((y == 1) & (pred == 0)).sum())
        fake_as_real = int(((y == 0) & (pred == 1)).sum())
        rows.append({
            "threshold": t,
            "real_as_fake": real_as_fake,
            "fake_as_real": fake_as_real,
            "cost": COST_REAL_AS_FAKE * real_as_fake + COST_FAKE_AS_REAL * fake_as_real,
            "f1": f1_score(y, pred, zero_division=0),
        })
    table = pd.DataFrame(rows)
    best = table[table["cost"] == table["cost"].min()].copy()
    best["distance_from_0.5"] = (best["threshold"] - 0.5).abs()
    threshold = float(best.sort_values("distance_from_0.5").iloc[0]["threshold"])
    return threshold, table


def compute_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "real_called_fake": int(fn),
        "fake_called_real": int(fp),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_threshold_search(name, title, table, threshold):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["threshold"], table["real_as_fake"], label="Real called fake")
    ax.plot(table["threshold"], table["fake_as_real"], label="Fake called real")
    ax.plot(table["threshold"], table["cost"], label="Weighted cost", linewidth=2)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Chosen threshold ({threshold:.2f})")
    ax.set_xlabel("Threshold on P(Real)")
    ax.set_ylabel("Count on validation set")
    ax.set_title(f"{title}: threshold search on validation data")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_threshold_search.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = confusion_matrix(r["y"], (r["p"] >= r["threshold"]).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(f"{MODELS[name]['title']}\n(test set, threshold {r['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        fpr, tpr, thr = roc_curve(r["y"], r["p"])
        auc = roc_auc_score(r["y"], r["p"])
        line, = ax.plot(fpr, tpr, label=f"{MODELS[name]['title']} (AUC = {auc:.3f})")
        idx = int(np.argmin(np.abs(thr - r["threshold"])))
        ax.scatter(fpr[idx], tpr[idx], color=line.get_color(), zorder=3)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (fake called real)")
    ax.set_ylabel("True positive rate (real correctly called real)")
    ax.set_title("ROC curves on the test set")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def save_misclassified(name, r, n=10):
    pred = (r["p"] >= r["threshold"]).astype(int)
    wrong = pred != r["y"]
    df = pd.DataFrame({
        "filepath": r["paths"][wrong],
        "true_class": [CLASS_NAMES[i] for i in r["y"][wrong]],
        "predicted_class": [CLASS_NAMES[i] for i in pred[wrong]],
        "p_real": np.round(r["p"][wrong], 4),
    })
    df["confidence_in_wrong_answer"] = np.where(
        df["predicted_class"] == "RealArt", df["p_real"], 1 - df["p_real"]
    ).round(4)
    df = df.sort_values("confidence_in_wrong_answer", ascending=False)
    df.to_csv(OUT_DIR / f"{name}_misclassified.csv", index=False)

    show = df.head(n)
    if show.empty:
        return 0
    cols = 5
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (ax, row) in enumerate(zip(axes, show.itertuples())):
        with Image.open(row.filepath) as img:
            size = f"{img.width}x{img.height}"
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"#{i + 1} True: {row.true_class}\nPred: {row.predicted_class} "
            f"(P(Real)={row.p_real:.2f})\n{size}",
            fontsize=9,
        )
    fig.suptitle(f"{MODELS[name]['title']}: most confident mistakes on the test set")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_misclassified_grid.png", dpi=150)
    plt.close(fig)
    return len(df)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []

    for name, cfg in MODELS.items():
        model = tf.keras.models.load_model(MODEL_DIR / cfg["file"], compile=False)

        _, y_val, p_val = predict(model, "val", cfg["normalize"])
        threshold, search = choose_threshold(y_val, p_val)
        search.to_csv(OUT_DIR / f"{name}_threshold_search.csv", index=False)
        plot_threshold_search(name, cfg["title"], search, threshold)

        paths, y_test, p_test = predict(model, "test", cfg["normalize"])
        results[name] = {"paths": paths, "y": y_test, "p": p_test, "threshold": threshold}

        rows.append({"model": cfg["title"], "split": "val", **compute_metrics(y_val, p_val, threshold)})
        rows.append({"model": cfg["title"], "split": "test (threshold 0.5)",
                     **compute_metrics(y_test, p_test, 0.5)})
        rows.append({"model": cfg["title"], "split": "test (chosen threshold)",
                     **compute_metrics(y_test, p_test, threshold)})

        pd.DataFrame({
            "filepath": paths, "true_label": y_test, "p_real": np.round(p_test, 4),
            "predicted_label": (p_test >= threshold).astype(int),
        }).to_csv(OUT_DIR / f"{name}_test_predictions.csv", index=False)

        n_wrong = save_misclassified(name, results[name])
        print(f"{cfg['title']}: threshold {threshold:.2f}, {n_wrong} test mistakes")

    plot_confusion_matrices(results)
    plot_roc_curves(results)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "metrics.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(json.dumps(rows, indent=2))

    cols = ["model", "split", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
            "real_called_fake", "fake_called_real"]
    print("\n" + table[cols].to_string(index=False))
    print(f"\nSaved everything to {OUT_DIR}/")


if __name__ == "__main__":
    main()
