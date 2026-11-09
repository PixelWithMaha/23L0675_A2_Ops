import time

import requests


class ModelAPIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


class RemoteClassifier:
    def __init__(self, url, api_key, timeout=30):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.model_name, self.model_version, self.threshold = "visionguard-classifier@vercel", "unknown", None

        start = time.perf_counter()
        info = self.health()
        if info:
            self.model_name = f"{info['model_name']}@vercel"
            self.model_version = str(info["model_version"])
            self.threshold = info["threshold"]
        self.load_seconds = time.perf_counter() - start

    def health(self):
        try:
            response = requests.get(f"{self.url}/health", timeout=self.timeout)
            return response.json() if response.ok else None
        except requests.RequestException:
            return None

    def predict(self, image_bytes, filename="upload.jpg"):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.url}/predict",
                headers={"X-API-Key": self.api_key},
                files={"image": (filename, image_bytes)},
                timeout=self.timeout,
            )
        except requests.Timeout:
            raise ModelAPIError(504, "The model API took too long to answer. Please try again.")
        except requests.RequestException:
            raise ModelAPIError(503, "The model API could not be reached. Check the internet connection.")

        if not response.ok:
            try:
                message = response.json().get("error", response.text)
            except ValueError:
                message = response.text[:200]
            raise ModelAPIError(response.status_code, message)

        result = response.json()
        self.model_version = str(result["model_version"])
        self.threshold = result["threshold"]
        return {
            "label": result["label"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "p_real": result["p_real"],
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
