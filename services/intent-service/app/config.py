"""Intent Classification Service Configuration"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model configuration
    model_name: str = os.getenv("MODEL_NAME", "claudiomello/AcademiCK-intent-classifier")

    # Device configuration - default to CPU for this lightweight model
    device: str = os.getenv("DEVICE", "cpu")

    class Config:
        env_file = ".env"


settings = Settings()
