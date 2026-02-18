from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=120,
        imgsz=640,
        batch=8,
        device=0,
        amp=True,
        cos_lr=True,
        patience=30,
        optimizer="AdamW",
        workers=4,
        cache=True,
        close_mosaic=15,
        name="pcba_core10_v14"
    )

if __name__ == "__main__":
    main()
