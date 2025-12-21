# app/config.py - UPDATED WITH SMTP SETTINGS

import logging
from pathlib import Path
from typing import List

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Server
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "SeeDeep.AI"
    VERSION: str = "2.0.0"

    # Database
    DATABASE_URL: str = "mysql+aiomysql://root:behzaD!1@localhost:3306/seedeep"

    # CORS
    ALLOWED_ORIGINS: str = "*"

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def parse_cors(cls, v: str) -> List[str]:
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]

    # Models
    MODEL_DIR: Path = Path("app/models/weights")
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    MAX_DETECTIONS: int = 100
    FORCE_CPU: bool = False

    # Device
    @property
    def DEVICE(self) -> torch.device:
        return torch.device(
            "cuda" if torch.cuda.is_available() and not self.FORCE_CPU else "cpu"
        )

    # Available Models
    AVAILABLE_MODELS: dict = {
        "face_detection": "Facemask.pt",
        "cap_detection": "Cap.pt",
        "weapon_detection": "Weapon.pt",
        "fire_detection": "Fire.pt",
        "general_detection": "YOLO.pt",
    }

    # SMTP Settings for Email Alerts
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USE_TLS: bool = True
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    SMTP_FROM_NAME: str = "SeeDeep.AI Alerts"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

    def validate_models(self):
        """Check which model files actually exist"""
        missing = []
        available = []

        for model_name, model_file in self.AVAILABLE_MODELS.items():
            model_path = self.MODEL_DIR / model_file
            if model_path.exists():
                available.append(model_name)
                logger.info(f"✅ Model found: {model_name} ({model_file})")
            else:
                missing.append(f"{model_name} ({model_file})")
                logger.warning(f"❌ Model missing: {model_name} ({model_file})")

        if missing:
            logger.warning(f"⚠️ Missing models: {', '.join(missing)}")
            logger.info(f"💡 Place model files in: {self.MODEL_DIR.absolute()}")

        return available

    def validate_smtp(self):
        """Validate SMTP configuration"""
        if not self.SMTP_USERNAME or not self.SMTP_PASSWORD:
            logger.warning(
                "⚠️ SMTP credentials not configured - email alerts will be disabled"
            )
            return False

        if not self.SMTP_FROM_EMAIL:
            logger.warning("⚠️ SMTP_FROM_EMAIL not configured - using SMTP_USERNAME")
            self.SMTP_FROM_EMAIL = self.SMTP_USERNAME

        logger.info(f"✅ SMTP configured: {self.SMTP_SERVER}:{self.SMTP_PORT}")
        logger.info(f"   From: {self.SMTP_FROM_EMAIL}")
        return True


settings = Settings()

# Log model availability on startup
logger.info(f"📂 Model directory: {settings.MODEL_DIR.absolute()}")
available_models = settings.validate_models()
logger.info(f"✅ Available models: {available_models}")

# Validate SMTP
settings.validate_smtp()
