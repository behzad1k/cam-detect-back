from ultralytics import YOLO

model_path = 'app/models/weights/Smoke.pt'
test_path = 'app/data/9.mp4'
model = YOLO(model_path)

result = model.predict(source=test_path, save=True)