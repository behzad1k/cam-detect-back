import os
from typing import Dict, List
from pathlib import Path
import torch


class Settings:
  # API Configuration
  TITLE: str = "SeeDeep.Ai"
  ROOT_PATH: str = "/cam-detection"
  DESCRIPTION: str = "Real-time object detection by SeeDeep.Ai"
  VERSION: str = "1.0.0"
  # DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
  DEBUG: bool = "true"

  # Server Configuration`
  HOST: str = os.getenv("HOST", "0.0.0.0")
  PORT: int = int(os.getenv("PORT", "8000"))
  RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"

  # CORS Configuration
  ALLOWED_ORIGINS: List[str] = ["*"]
  # ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

  # Model Configuration
  CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
  IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
  MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", "100"))
  FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"

  # Device Configuration
  DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
  )

  # Model Paths
  MODEL_PATHS: Dict[str, str] = {
    "face_detection": "app/models/weights/Facemask.onnx",
    "cap_detection": "app/models/weights/Cap.onnx",
    "ppe_detection": "app/models/weights/PPE.onnx",
  }
  MODEL_CLASSES = {
    "cap_detection": {
      0: "no_cap",
      1: "cap",
    },
    "face_detection": {
      0: "no_mask",
      1: "mask",
    },
    "ppe_detection": {
      0: "person",
      1: "ear",
      2: "ear-mufs",
      3: "face",
      4: "face-guard",
      5: "face-mask",
      6: "foot",
      7: "tool",
      8: "glasses",
      9: "gloves",
      10: "helmet",
      11: "hands",
      12: "head",
      13: "medical-suit",
      14: "shoes",
      15: "safety-suit",
      16: "safety-vest",
    }
  }


settings = Settings()
