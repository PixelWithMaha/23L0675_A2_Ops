"""VisionGuard Flask app: upload, result, history and health pages.

Run locally (debug off):  python -m app.main
"""
import base64
import hashlib
import io

from flask import Flask, abort, jsonify, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from app.config import Config
from app.db import Database
from app.remote import ModelAPIError

DECODABLE_FORMATS = {"JPEG", "PNG", "BMP", "GIF"}


def wants_json():
    return request.headers.get("X-Requested-With") == "fetch"


def fail(status, title, message):
    if wants_json():
        return jsonify({"title": title, "message": message}), status
    return render_template("error.html", title=title, message=message), status


class UploadError(Exception):
    def __init__(self, status, title, message):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


def validate_upload(file):
    """Return (safe_filename, image_bytes) or raise UploadError. Nothing is written to disk."""
    if file is None or not file.filename:
        raise UploadError(400, "No file selected", "Please choose an image to upload.")

    filename = secure_filename(file.filename) or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in Config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(Config.ALLOWED_EXTENSIONS))
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Allowed types: {allowed}.")

    data = file.read()  # kept in memory only
    if not data:
        raise UploadError(400, "Empty file", "The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
            fmt = img.format
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadError(400, "Image could not be read",
                          "The file has an image extension but its contents are not a valid image. "
                          "It may be corrupted or renamed from another file type.")
    if fmt not in DECODABLE_FORMATS:
        raise UploadError(415, "Unsupported image format", f"{fmt} images are not supported.")
    return filename, data


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    if Config.INFERENCE_MODE == "remote":
        from app.remote import RemoteClassifier
        classifier = RemoteClassifier(Config.MODEL_API_URL, Config.MODEL_API_KEY, Config.MODEL_API_TIMEOUT)
    else:
        from app.inference import Classifier
        classifier = Classifier(Config.MODEL_PATH, Config.MODEL_META_PATH, Config.THRESHOLD)
    db = Database(Config.DATABASE_URL)
    app.logger.info("Inference mode %s: model %s v%s ready in %.1fs", Config.INFERENCE_MODE,
                    classifier.model_name, classifier.model_version, classifier.load_seconds)

    @app.context_processor
    def inject_model():
        return {"model_name": classifier.model_name, "model_version": classifier.model_version}

    @app.route("/", methods=["GET"])
    def upload():
        return render_template("upload.html", max_mb=Config.MAX_UPLOAD_MB,
                               allowed=sorted(Config.ALLOWED_EXTENSIONS))

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            filename, data = validate_upload(request.files.get("image"))  # may raise 413 (too large)
        except UploadError as err:
            return fail(err.status, err.title, err.message)
        try:
            result = classifier.predict(data, filename)
        except ModelAPIError as err:
            app.logger.error("Model API error %s: %s", err.status, err.message)
            status = err.status if 400 <= err.status < 500 else 502
            return fail(status, "Model API error", err.message)
        except Exception:
            app.logger.exception("Prediction failed")
            message = "The model could not read this image. Please try another file."
            return fail(400, "Image could not be decoded", message)

        row = db.add(
            filename=filename,
            image_sha256=hashlib.sha256(data).hexdigest(),
            label=result["label"],
            confidence=result["confidence"],
            p_real=result["p_real"],
            threshold=classifier.threshold,
            model_name=classifier.model_name,
            model_version=classifier.model_version,
            latency_ms=result["latency_ms"],
        )
        if wants_json():
            return jsonify({"id": row.id, "url": url_for("result", prediction_id=row.id, fresh=1)})
        preview = f"data:image/*;base64,{base64.b64encode(data).decode()}"  # shown once, never saved
        return render_template("result.html", p=row, preview=preview, fresh=True)

    @app.route("/result/<int:prediction_id>", methods=["GET"])
    def result(prediction_id):
        row = db.get(prediction_id)
        if row is None:
            abort(404)
        return render_template("result.html", p=row, preview=None, fresh=request.args.get("fresh") == "1")

    @app.route("/history", methods=["GET"])
    def history():
        return render_template("history.html", rows=db.latest(100))

    @app.route("/health", methods=["GET"])
    def health():
        db_ok = db.is_up()
        model_ok = Config.INFERENCE_MODE != "remote" or classifier.health() is not None
        body = {
            "status": "ok" if db_ok and model_ok else "degraded",
            "inference_mode": Config.INFERENCE_MODE,
            "model_loaded": model_ok,
            "model_name": classifier.model_name,
            "model_version": classifier.model_version,
            "threshold": classifier.threshold,
            "database": "ok" if db_ok else "unreachable",
        }
        if Config.INFERENCE_MODE == "remote":
            body["model_api"] = Config.MODEL_API_URL if model_ok else "unreachable"
        return jsonify(body), 200 if db_ok and model_ok else 503

    @app.errorhandler(HTTPException)
    def http_error(err):
        messages = {
            404: ("Page not found", "That page does not exist."),
            405: ("Method not allowed", "Please use the upload form."),
            413: ("File too large", f"Images must be {Config.MAX_UPLOAD_MB:g} MB or smaller."),
        }
        title, message = messages.get(err.code, (err.name, err.description))
        return fail(err.code, title, message)

    @app.errorhandler(Exception)
    def server_error(err):
        app.logger.exception("Unhandled error")
        return fail(500, "Something went wrong", "An unexpected error occurred. Please try again.")

    return app


app = create_app()

if __name__ == "__main__":
    # Debug stays OFF: no interactive debugger (remote code execution risk), no auto-reloader
    # (which would load the model twice), and no stack traces shown to users.
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
