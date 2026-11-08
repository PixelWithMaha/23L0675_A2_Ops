"""Download the champion model from the MLflow registry ONCE and save it next to the app.

The Flask app then loads models/champion.keras from disk at start-up, so it never has to
call DagsHub while serving requests.

    python scripts/export_champion.py
"""
import json
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

NAME = os.getenv("REGISTERED_MODEL_NAME", "visionguard-classifier")
ALIAS = os.getenv("REGISTERED_MODEL_ALIAS", "champion")
OUT_DIR = Path("models")


def main():
    client = MlflowClient()
    version = client.get_model_version_by_alias(NAME, ALIAS)
    run = client.get_run(version.run_id)
    model = mlflow.keras.load_model(f"models:/{NAME}@{ALIAS}")

    OUT_DIR.mkdir(exist_ok=True)
    model.save(OUT_DIR / "champion.keras")
    meta = {
        "model_name": NAME,
        "alias": ALIAS,
        "version": version.version,
        "run_id": version.run_id,
        "run_name": run.info.run_name,
        "model_type": run.data.params.get("model", "transfer"),
        "image_size": int(model.input_shape[1]),
        "test_auc": run.data.metrics.get("test_auc"),
    }
    (OUT_DIR / "champion.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {OUT_DIR / 'champion.keras'} and champion.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
