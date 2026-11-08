"""Task 3.1 / 3.2: tiny Keras model on Fashion-MNIST tracked with MLflow.

Local (3.1):   python src/mlflow_toy.py --lr 0.001
DagsHub (3.2): put MLFLOW_TRACKING_URI / USERNAME / PASSWORD in .env, then run the same command.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

# Credentials come from .env (git-ignored), never from the code.
load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "toy-fashion-mnist"
SEED = 42


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Sends every epoch's metrics to MLflow so we get curves, not just final numbers."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def build_model(units, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_curves(history, lr):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, h[metric], marker="o", label=f"train {metric}")
        ax.plot(epochs, h[f"val_{metric}"], marker="o", label=f"val {metric}")
        ax.set_xlabel("Epoch")
        ax.set_title(metric.capitalize())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Fashion-MNIST, learning rate {lr}")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--units", type=int, default=128)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    where = "dagshub" if "dagshub" in TRACKING_URI else "local"
    print(f"Tracking to: {TRACKING_URI} ({where})")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    with mlflow.start_run(run_name=f"lr={args.lr}") as run:
        mlflow.set_tags({"task": "3.1-3.2 toy", "dataset": "fashion_mnist", "tracking": where})
        mlflow.log_params({
            "learning_rate": args.lr, "epochs": args.epochs, "units": args.units,
            "batch_size": 128, "optimizer": "adam", "seed": SEED,
        })

        model = build_model(args.units, args.lr)
        history = model.fit(
            x_train, y_train, validation_split=0.2, epochs=args.epochs,
            batch_size=128, callbacks=[MLflowEpochLogger()], verbose=2,
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        fig = plot_curves(history, args.lr)
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)

        mlflow.keras.log_model(model, name="model")

        print(f"Run {run.info.run_id} finished: test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
