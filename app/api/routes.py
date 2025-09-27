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


from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, List, Optional
import json
import os

from ..core.calibration import CameraCalibration, CalibrationMode, CalibrationPoint
from ..core.schemas import CalibrationRequest

router = APIRouter()

# Global calibration instance (in production, use dependency injection)
calibration_manager = CameraCalibration()


@router.post("/calibration/setup")
async def setup_calibration(request: CalibrationRequest):
  """Setup camera calibration"""
  try:
    # Convert points
    points = [
      CalibrationPoint(
        pixel_x=p.pixel_x,
        pixel_y=p.pixel_y,
        real_x=p.real_x,
        real_y=p.real_y
      )
      for p in request.points
    ]

    # Perform calibration based on mode
    success = False
    if request.mode == "perspective_transform":
      success = calibration_manager.set_perspective_transform_calibration(
        points, request.frame_width, request.frame_height
      )
    elif request.mode == "reference_object":
      success = calibration_manager.set_reference_object_calibration(
        points,
        request.reference_width_meters,
        request.reference_height_meters,
        request.frame_width,
        request.frame_height
      )
    elif request.mode == "vanishing_point":
      success = calibration_manager.set_vanishing_point_calibration(
        tuple(request.vanishing_point),
        request.horizon_line,
        request.reference_distance_meters,
        request.reference_pixel_height,
        request.frame_width,
        request.frame_height
      )

    return {
      "success": success,
      "calibration_info": calibration_manager.get_calibration_info()
    }

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Calibration failed: {str(e)}")


@router.get("/calibration/info")
async def get_calibration_info():
  """Get current calibration information"""
  return calibration_manager.get_calibration_info()


@router.post("/calibration/save")
async def save_calibration(filepath: str = "calibration.json"):
  """Save current calibration to file"""
  success = calibration_manager.save_calibration(filepath)
  if not success:
    raise HTTPException(status_code=500, detail="Failed to save calibration")

  return {"success": True, "filepath": filepath}


@router.post("/calibration/load")
async def load_calibration(filepath: str = "calibration.json"):
  """Load calibration from file"""
  if not os.path.exists(filepath):
    raise HTTPException(status_code=404, detail="Calibration file not found")

  success = calibration_manager.load_calibration(filepath)
  if not success:
    raise HTTPException(status_code=500, detail="Failed to load calibration")

  return {
    "success": True,
    "filepath": filepath,
    "calibration_info": calibration_manager.get_calibration_info()
  }


@router.post("/calibration/upload")
async def upload_calibration(file: UploadFile = File(...)):
  """Upload and load calibration file"""
  if not file.filename.endswith('.json'):
    raise HTTPException(status_code=400, detail="Only JSON files are supported")

  try:
    content = await file.read()
    calibration_data = json.loads(content)

    # Save to temporary file and load
    temp_filepath = f"temp_{file.filename}"
    with open(temp_filepath, 'w') as f:
      json.dump(calibration_data, f)

    success = calibration_manager.load_calibration(temp_filepath)

    # Clean up temp file
    os.remove(temp_filepath)

    if not success:
      raise HTTPException(status_code=400, detail="Invalid calibration file format")

    return {
      "success": True,
      "filename": file.filename,
      "calibration_info": calibration_manager.get_calibration_info()
    }

  except json.JSONDecodeError:
    raise HTTPException(status_code=400, detail="Invalid JSON file")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/calibration/convert-point")
async def convert_point(pixel_x: float, pixel_y: float):
  """Convert pixel coordinates to real-world meters"""
  if not calibration_manager.is_calibrated:
    raise HTTPException(status_code=400, detail="Camera not calibrated")

  real_x, real_y = calibration_manager.pixel_to_meters(pixel_x, pixel_y)

  return {
    "pixel_coordinates": [pixel_x, pixel_y],
    "real_world_coordinates": [round(real_x, 3), round(real_y, 3)],
    "units": "meters"
  }


@router.post("/calibration/calculate-distance")
async def calculate_distance(
  point1_x: float, point1_y: float,
  point2_x: float, point2_y: float
):
  """Calculate real-world distance between two pixel points"""
  if not calibration_manager.is_calibrated:
    raise HTTPException(status_code=400, detail="Camera not calibrated")

  distance = calibration_manager.calculate_distance(
    (point1_x, point1_y), (point2_x, point2_y)
  )

  return {
    "point1": [point1_x, point1_y],
    "point2": [point2_x, point2_y],
    "distance_meters": round(distance, 3),
    "distance_feet": round(distance * 3.28084, 3)
  }
