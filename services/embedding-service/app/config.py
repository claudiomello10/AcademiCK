"""Embedding Service Configuration"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model configuration
    model_name: str = os.getenv("MODEL_NAME", "BAAI/bge-m3")
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    max_length: int = int(os.getenv("MAX_LENGTH", "8192"))

    # Device configuration
    device: str = os.getenv("DEVICE", "gpu")  # "cpu" or "gpu"

    # GPU configuration
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    use_fp16: bool = os.getenv("USE_FP16", "true").lower() == "true"

    class Config:
        env_file = ".env"


settings = Settings()
