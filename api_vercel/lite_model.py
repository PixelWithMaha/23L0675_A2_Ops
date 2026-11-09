import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

CLASS_NAMES = ["AiArtData", "RealArt"]
LABELS = {"AiArtData": "AI-generated", "RealArt": "Real"}


def decode(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.seek(0)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


class LiteClassifier:
    def __init__(self, model_path, meta_path=None, threshold=0.5):
        start = time.perf_counter()
        self.interpreter = Interpreter(model_path=str(model_path))
        self.input = self.interpreter.get_input_details()[0]["index"]
        self.output = self.interpreter.get_output_details()[0]["index"]
        self.meta = json.loads(Path(meta_path).read_text()) if meta_path and Path(meta_path).exists() else {}
        self.threshold = threshold
        self.model_name = self.meta.get("model_name", "visionguard-classifier")
        self.model_version = str(self.meta.get("version", "local"))
        self.load_seconds = time.perf_counter() - start

    def p_real(self, pixels):
        x = pixels[np.newaxis].astype(np.float32)
        self.interpreter.resize_tensor_input(self.input, x.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input, x)
        self.interpreter.invoke()
        return float(self.interpreter.get_tensor(self.output)[0][0])

    def predict(self, image_bytes):
        start = time.perf_counter()
        p_real = self.p_real(decode(image_bytes))
        is_real = p_real >= self.threshold
        label = CLASS_NAMES[int(is_real)]
        return {
            "label": label,
            "verdict": LABELS[label],
            "confidence": p_real if is_real else 1 - p_real,
            "p_real": p_real,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
