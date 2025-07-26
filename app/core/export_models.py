import argparse
from pathlib import Path
from ultralytics import YOLO

def export_model(model_path: str, output_dir: str = "exported_models", imgsz: int = 640):
  """Export a YOLO model to ONNX format"""
  model = YOLO(model_path)

  # Create output directory if it doesn't exist
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  # Export to ONNX
  output_path = Path(output_dir) / Path(model_path).with_suffix('.onnx').name
  model.export(
    format='onnx',
    # imgsz=imgsz,
    dynamic=False,
    # simplify=False,
    opset=12,
    nms=True,
    batch=1
  )

  # Move the exported file to our output directory
  default_export_path = Path(model_path).with_suffix('.onnx')
  if default_export_path.exists():
    default_export_path.rename(output_path)

  print(f"Model exported to: {output_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Export YOLO models to ONNX format')
  parser.add_argument('model_path', type=str, help='Path to the .pt model file')
  parser.add_argument('--output-dir', type=str, default="exported_models", help='Output directory for exported models')
  parser.add_argument('--imgsz', type=int, default=640, help='Input image size (square)')

  args = parser.parse_args()
  export_model(args.model_path, args.output_dir, args.imgsz)