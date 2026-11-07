"""Task 2.7 toy part: tune a tiny network on Fashion-MNIST with Keras Tuner (Hyperband)."""
from pathlib import Path

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

SEED = 42
OUT_DIR = Path("outputs/tuner")
TUNER_DIR = Path("tuner_runs")


def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hp.Int("units", min_value=32, max_value=512, step=32), activation="relu"
        ),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def trials_table(tuner):
    rows = []
    for trial in tuner.oracle.trials.values():
        if trial.score is None:
            continue
        rows.append({
            "trial": trial.trial_id,
            **trial.hyperparameters.values,
            "val_accuracy": round(trial.score, 4),
        })
    keep = ["trial", "units", "learning_rate", "tuner/epochs", "val_accuracy"]
    df = pd.DataFrame(rows)
    return df[[c for c in keep if c in df.columns]].sort_values("val_accuracy", ascending=False)


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=6,
        factor=3,
        seed=SEED,
        directory=TUNER_DIR,
        project_name="fashion_mnist_toy",
        overwrite=True,
    )
    tuner.search_space_summary()
    tuner.search(
        x_train, y_train, validation_split=0.2, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
    )

    table = trials_table(tuner)
    table.to_csv(OUT_DIR / "toy_fashion_mnist_trials.csv", index=False)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    _, test_acc = best_model.evaluate(x_test, y_test, verbose=0)

    print("\nAll trials (best first):")
    print(table.to_string(index=False))
    print("\nBest settings:")
    print(f"  units         = {best_hp.get('units')}")
    print(f"  learning_rate = {best_hp.get('learning_rate')}")
    print(f"  test accuracy of best model = {test_acc:.4f}")


if __name__ == "__main__":
    main()
