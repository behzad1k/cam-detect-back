import argparse
from pathlib import Path
import onnxruntime as ort
from config import settings


def verify_model(model_name: str):
  """Verify a model is properly configured and loadable"""
  print(f"\nVerifying model: {model_name}")

  # Check configuration
  if model_name not in settings.MODEL_PATHS:
    print(f"❌ Model not found in MODEL_PATHS configuration")
    return False

  model_path = settings.MODEL_PATHS[model_name]
  if not Path(model_path).exists():
    print(f"❌ Model file not found at: {model_path}")
    return False

  # Check ONNX model
  try:
    session = ort.InferenceSession(model_path)
    print(f"✅ Model loaded successfully")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Outputs: {[out.name for out in session.get_outputs()]}")

    # Check classes
    if model_name in settings.MODEL_CLASSES:
      print(f"Classes: {len(settings.MODEL_CLASSES[model_name])} classes configured")
    else:
      print("⚠️ No classes configured")

    return True
  except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    return False


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Verify model configurations')
  parser.add_argument('model_name', type=str, help='Name of the model to verify')
  args = parser.parse_args()

  if args.model_name == "all":
    for model_name in settings.MODEL_PATHS.keys():
      verify_model(model_name)
  else:
    verify_model(args.model_name)