"""Training and serving must preprocess images identically."""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import parse_image  # noqa: E402  (training code)

from app.inference import preprocess  # noqa: E402  (serving code)


def make_image(path, fmt, mode="RGB", size=(300, 200)):
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    arr = (rng.random((size[1], size[0], channels)) * 255).astype("uint8").squeeze()
    Image.fromarray(arr, mode=mode).save(path, format=fmt)
    return path


def test_serving_matches_training(tmp_path):
    for name, fmt, mode in [("a.jpg", "JPEG", "RGB"), ("b.png", "PNG", "RGBA"), ("c.png", "PNG", "L")]:
        path = make_image(tmp_path / name, fmt, mode)
        for size, normalize in [((160, 160), False), ((224, 224), True)]:
            train_img, _ = parse_image(tf.constant(str(path)), 0, normalize=normalize, image_size=size)
            serve_img = preprocess(path.read_bytes(), size, normalize)
            assert serve_img.shape == (*size, 3)
            np.testing.assert_allclose(train_img.numpy(), serve_img.numpy(), atol=1e-5)
