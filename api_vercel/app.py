import hmac
import io
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lite_model import LiteClassifier  # noqa: E402

API_KEY = os.getenv("API_KEY", "")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "4"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)

classifier = LiteClassifier(ROOT / "model" / "champion.tflite", ROOT / "model" / "champion.json", THRESHOLD)


def error(status, message):
    return jsonify({"error": message}), status


@app.route("/", methods=["GET"])
def index():
    endpoints = {"GET /health": "status", "POST /predict": "form field 'image', header X-API-Key"}
    return jsonify({"service": "VisionGuard model API (23L0675_A2_Ops)", "endpoints": endpoints})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "model_name": classifier.model_name,
                    "model_version": classifier.model_version, "runtime": "tflite",
                    "threshold": classifier.threshold, "max_upload_mb": MAX_UPLOAD_MB})


@app.route("/predict", methods=["POST"])
def predict():
    if not API_KEY:
        return error(500, "API_KEY is not configured on the server.")
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return error(401, "Missing or wrong API key (send it in the X-API-Key header).")

    file = request.files.get("image")
    if file is None or not file.filename:
        return error(400, "Send the image in a form field called 'image'.")
    filename = secure_filename(file.filename) or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return error(415, f"'.{ext or '?'}' files are not accepted. Allowed: {allowed}.")

    data = file.read()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        result = classifier.predict(data)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return error(400, "The file is not a valid image (corrupted or renamed).")
    return jsonify({"filename": filename, **result})


@app.errorhandler(HTTPException)
def http_error(err):
    if err.code == 413:
        return error(413, f"Image too large. Maximum is {MAX_UPLOAD_MB:g} MB.")
    if err.code == 404:
        return jsonify({"error": "Not found", "path": request.path}), 404
    return error(err.code, err.description)
