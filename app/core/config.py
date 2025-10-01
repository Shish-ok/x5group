from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    app_name: str = "NER API"

    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "3"))
    max_workers: int = max(2, (os.cpu_count() or 2) * 2)

    use_transformers: bool = Field(default_factory=lambda: os.getenv("USE_TRANSFORMERS", "0") == "1")
    model_path: str | None = os.getenv("MODEL_PATH")
    model_name: str | None = os.getenv("MODEL_NAME")
    device: str | None = os.getenv("DEVICE")

settings = Settings()