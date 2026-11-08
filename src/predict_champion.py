"""Task 3.6: load the model by its registry alias and classify one image.

    python src/predict_champion.py path/to/image.jpg
"""
import argparse
import os

import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient

from dataset import parse_image

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
CLASS_NAMES = ["AiArtData", "RealArt"]  # sigmoid output = P(RealArt)
THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--name", default="visionguard-classifier")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    client = MlflowClient()
    version = client.get_model_version_by_alias(args.name, args.alias)
    run = client.get_run(version.run_id)
    print(f"Loading models:/{args.name}@{args.alias}  ->  version {version.version} "
          f"(run '{run.info.run_name}')")

    model = mlflow.keras.load_model(f"models:/{args.name}@{args.alias}")

    # Same preprocessing as training: decode, 3 channels, resize to the model's input size.
    # The transfer model rescales pixels itself; the baseline CNN expects 0-1 input.
    image_size = tuple(model.input_shape[1:3])
    normalize = run.data.params.get("model") == "baseline"
    img, _ = parse_image(tf.constant(args.image), 0, normalize=normalize, image_size=image_size)

    p_real = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0][0])
    label = CLASS_NAMES[int(p_real >= THRESHOLD)]
    confidence = p_real if label == "RealArt" else 1 - p_real
    print(f"Image      : {args.image}")
    print(f"Input size : {image_size}")
    print(f"P(RealArt) : {p_real:.4f}")
    print(f"Prediction : {label} (confidence {confidence:.1%})")


if __name__ == "__main__":
    main()
