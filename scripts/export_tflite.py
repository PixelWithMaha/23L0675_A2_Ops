import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api_vercel"))
from dataset import parse_image, read_split  # noqa: E402
from lite_model import LiteClassifier  # noqa: E402

KERAS_PATH = ROOT / "models" / "champion.keras"
META_PATH = ROOT / "models" / "champion.json"
OUT_DIR = ROOT / "api_vercel" / "model"


def convert():
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    size = tuple(model.input_shape[1:3])

    inputs = tf.keras.Input(shape=(None, None, 3), batch_size=1, name="image")
    resized = tf.keras.layers.Resizing(*size, interpolation="bilinear")(inputs)
    outputs = model(resized, training=False)
    wrapper = tf.keras.Model(inputs, outputs)

    with tempfile.TemporaryDirectory() as tmp:
        wrapper.export(tmp, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        tflite_bytes = converter.convert()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "champion.tflite").write_bytes(tflite_bytes)
    shutil.copy(META_PATH, OUT_DIR / "champion.json")
    size_mb = (OUT_DIR / "champion.tflite").stat().st_size / 1e6
    print(f"Saved {OUT_DIR / 'champion.tflite'} ({size_mb:.1f} MB)")
    return model, size


def check(model, size):
    lite = LiteClassifier(OUT_DIR / "champion.tflite", OUT_DIR / "champion.json")
    paths, _ = read_split("test")
    diffs, agree = [], 0
    for path in paths:
        img, _ = parse_image(tf.constant(str(ROOT / path)), 0, normalize=False, image_size=size)
        p_keras = float(model(img[tf.newaxis], training=False)[0][0])
        p_lite = lite.predict((ROOT / path).read_bytes())["p_real"]
        diffs.append(abs(p_keras - p_lite))
        agree += (p_keras >= 0.5) == (p_lite >= 0.5)
    print(f"Test images: {len(paths)} | same label: {agree}/{len(paths)} | "
          f"max |P(Real) difference|: {max(diffs):.6f} | mean: {np.mean(diffs):.6f}")


if __name__ == "__main__":
    check(*convert())
