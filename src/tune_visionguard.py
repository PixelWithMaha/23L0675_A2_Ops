"""Task 2.7 apply part: tune learning rate, dropout and number of unfrozen
MobileNetV2 blocks for VisionGuard with Keras Tuner (random search)."""
from pathlib import Path

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from dataset import SEED, get_data_splits
from model import compile_baseline_cnn
from transfer_model import build_transfer_model, unfreeze_last_block

OUT_DIR = Path("outputs/tuner")
TUNER_DIR = Path("tuner_runs")
STAGE1_MODEL = Path("models/transfer_stage1.keras")
MAX_TRIALS = 10
EPOCHS_PER_TRIAL = 6

# 0 = only the head trains; 1 = last block (block_16 + Conv_1); 2 = last two blocks; ...
UNFREEZE_FROM = {1: "block_16_expand", 2: "block_15_expand", 3: "block_14_expand"}

stage1_head = None
if STAGE1_MODEL.exists():
    stage1_head = tf.keras.models.load_model(STAGE1_MODEL, compile=False) \
        .get_layer("head_output").get_weights()


def build_model(hp):
    lr = hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])
    dropout = hp.Choice("dropout", [0.2, 0.3, 0.5])
    blocks = hp.Choice("unfrozen_blocks", [0, 1, 2, 3])

    model, base = build_transfer_model(dropout=dropout)
    if stage1_head is not None:
        # Warm start from the stage 1 head so every trial is a fine-tuning run
        model.get_layer("head_output").set_weights(stage1_head)
    if blocks > 0:
        unfreeze_last_block(base, UNFREEZE_FROM[blocks])
    return compile_baseline_cnn(model, learning_rate=lr)


def count_unfrozen_layers(blocks):
    if blocks == 0:
        return 0
    _, base = build_transfer_model()
    unfreeze_last_block(base, UNFREEZE_FROM[blocks])
    return sum(1 for layer in base.layers if layer.trainable and layer.weights)


def trials_table(tuner):
    rows = []
    for trial in tuner.oracle.trials.values():
        if trial.score is None:
            continue
        hp = trial.hyperparameters.values
        rows.append({
            "trial": trial.trial_id,
            "learning_rate": hp["learning_rate"],
            "dropout": hp["dropout"],
            "unfrozen_blocks": hp["unfrozen_blocks"],
            "unfrozen_layers": count_unfrozen_layers(hp["unfrozen_blocks"]),
            "val_loss": round(trial.score, 4),
            "val_accuracy": round(trial.metrics.get_best_value("val_accuracy"), 4),
            "val_auc": round(trial.metrics.get_best_value("val_auc"), 4),
        })
    return pd.DataFrame(rows).sort_values("val_loss")


def importance(table):
    lines = ["Effect of each setting (mean val_loss per value, lower is better):"]
    spreads = {}
    for name in ["learning_rate", "dropout", "unfrozen_blocks"]:
        means = table.groupby(name)["val_loss"].mean()
        spreads[name] = means.max() - means.min()
        values = ", ".join(f"{k}: {v:.4f}" for k, v in means.items())
        lines.append(f"  {name:16s} spread {spreads[name]:.4f}  ->  {values}")
    lines.append(f"Setting that mattered most: {max(spreads, key=spreads.get)}")
    return "\n".join(lines)


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, _ = get_data_splits(normalize=False)

    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=MAX_TRIALS,
        seed=SEED,
        directory=TUNER_DIR,
        project_name="visionguard_mobilenetv2",
        overwrite=True,
    )
    tuner.search_space_summary()
    tuner.search(
        train_ds, validation_data=val_ds, epochs=EPOCHS_PER_TRIAL,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )],
    )

    table = trials_table(tuner)
    table.to_csv(OUT_DIR / "visionguard_trials.csv", index=False)
    summary = importance(table)
    (OUT_DIR / "visionguard_importance.txt").write_text(summary + "\n")

    best = tuner.get_best_hyperparameters(1)[0].values
    print("\nAll trials (best first):")
    print(table.to_string(index=False))
    print("\n" + summary)
    print(f"\nBest settings: {best}")


if __name__ == "__main__":
    main()
