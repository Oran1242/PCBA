from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=8
)

#moth like pond