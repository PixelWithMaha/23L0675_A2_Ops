"""All settings come from environment variables (with safe defaults for local runs)."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    MODEL_META_PATH = os.getenv("MODEL_META_PATH", "models/champion.json")
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instance/visionguard.db")
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)  # Flask rejects bigger requests (413)
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5050"))
