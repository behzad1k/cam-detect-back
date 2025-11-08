from pathlib import Path
from typing import List
import torch
from pydantic_settings import BaseSettings
from pydantic import field_validator

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
  DATABASE_URL: str = "postgresql+asyncpg://seedeep:seedeep123@localhost:5432/seedeep_db"

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
  CONFIDENCE_THRESHOLD: float = 0.5
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
    # "ppe_detection": "PPE.pt",
    "weapon_detection": "Weapon.pt",
    "fire_detection": "Fire.pt",
    "general_detection": "YOLO.pt",
  }

  # Logging
  LOG_LEVEL: str = "INFO"

  class Config:
    case_sensitive = True
    env_file = ".env"
    extra = "ignore"


settings = Settings()