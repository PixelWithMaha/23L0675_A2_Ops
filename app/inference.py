"""Task 4.1: inference module.

The model is loaded ONCE when the app starts (load_model) and reused for every request.
Preprocessing repeats exactly what src/dataset.py did during training.

Try it from the terminal:  python -m app.inference image1.jpg image2.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["AiArtData", "RealArt"]  # sigmoid output = P(RealArt)
LABELS = {"AiArtData": "AI-generated", "RealArt": "Real"}


def preprocess(image_bytes, image_size, normalize=False):
    """Same steps as dataset.parse_image() in training: decode -> 3 channels -> resize (-> /255)."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    if normalize:
        img = img / 255.0
    return img


class Classifier:
    def __init__(self, model_path, meta_path=None, threshold=0.5):
        start = time.perf_counter()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.meta = json.loads(Path(meta_path).read_text()) if meta_path and Path(meta_path).exists() else {}
        self.threshold = threshold
        self.image_size = tuple(self.model.input_shape[1:3])
        # The transfer model rescales pixels itself; the baseline CNN was trained on 0-1 input.
        self.normalize = self.meta.get("model_type") == "baseline"
        self.model_name = self.meta.get("model_name", Path(model_path).stem)
        self.model_version = str(self.meta.get("version", "local"))
        self.model(np.zeros((1, *self.image_size, 3), dtype="float32"))  # warm-up call
        self.load_seconds = time.perf_counter() - start

    def preprocess(self, image_bytes):
        return tf.expand_dims(preprocess(image_bytes, self.image_size, self.normalize), 0)

    def predict(self, image_bytes):
        start = time.perf_counter()
        p_real = float(self.model(self.preprocess(image_bytes), training=False)[0][0])
        is_real = p_real >= self.threshold
        label = CLASS_NAMES[int(is_real)]
        return {
            "label": label,
            "verdict": LABELS[label],
            "confidence": p_real if is_real else 1 - p_real,
            "p_real": p_real,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


if __name__ == "__main__":
    from app.config import Config

    clf = Classifier(Config.MODEL_PATH, Config.MODEL_META_PATH, Config.THRESHOLD)
    print(f"Loaded {clf.model_name} v{clf.model_version} in {clf.load_seconds:.2f}s "
          f"(input {clf.image_size}, threshold {clf.threshold})")
    for path in sys.argv[1:]:
        r = clf.predict(Path(path).read_bytes())
        print(f"{path}\n  -> {r['verdict']} ({r['label']}), confidence {r['confidence']:.1%}, "
              f"P(Real)={r['p_real']:.4f}, {r['latency_ms']:.0f} ms")
