from typing import List, Dict
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.schemas import ModelInfo, HealthResponse
from app.models.model_manager import model_manager

router = APIRouter()


@router.get("/")
async def root():
  """Root endpoint with basic API information"""
  return {
    "message": "SeeDeep.Ai",
    "status": "running",
    "device": str(settings.DEVICE),
    "models_loaded": model_manager.get_loaded_models(),
    "debug": settings.DEBUG
  }


@router.get("/health", response_model=HealthResponse)
async def health_check():
  """Health check endpoint"""
  return HealthResponse(
    status="healthy",
    device=str(settings.DEVICE),
    models_available=model_manager.get_available_models(),
    models_loaded=model_manager.get_loaded_models()
  )


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
  """Get information about all available models"""
  models_info = []

  for model_name in model_manager.get_available_models():
    is_loaded = model_name in model_manager.get_loaded_models()
    available_classes = None

    if is_loaded:
      class_dict = model_manager.get_model_classes(model_name)
      if class_dict:
        available_classes = list(class_dict.values())

    models_info.append(ModelInfo(
      name=model_name,
      loaded=is_loaded,
      available_classes=available_classes,
      path=settings.MODEL_PATHS[model_name]
    ))

  return models_info


@router.get("/models/{model_name}/classes")
async def get_model_classes(model_name: str):
  """Get available class names for a specific model"""
  if model_name not in settings.MODEL_PATHS:
    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

  # Load model if not already loaded
  if model_name not in model_manager.get_loaded_models():
    if not model_manager.load_model(model_name):
      raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")

  class_dict = model_manager.get_model_classes(model_name)
  if class_dict is None:
    raise HTTPException(status_code=500, detail=f"Could not retrieve classes for model {model_name}")

  return {
    "model": model_name,
    "classes": class_dict,
    "class_names": list(class_dict.values())
  }


@router.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str):
  """Load a specific model"""
  if model_name not in settings.MODEL_PATHS:
    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

  success = model_manager.load_model(model_name)
  if success:
    class_dict = model_manager.get_model_classes(model_name)
    return {
      "success": True,
      "model": model_name,
      "device": str(settings.DEVICE),
      "available_classes": list(class_dict.values()) if class_dict else None
    }
  else:
    raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")


@router.post("/models/{model_name}/unload")
async def unload_model_endpoint(model_name: str):
  """Unload a specific model"""
  if model_name in model_manager.models:
    del model_manager.models[model_name]
    if model_name in model_manager.model_classes:
      del model_manager.model_classes[model_name]
    return {"success": True, "model": model_name, "status": "unloaded"}
  else:
    raise HTTPException(status_code=404, detail=f"Model {model_name} is not loaded")
