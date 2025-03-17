yolo \
    task=detect \
    mode=train \
    model=yolov8n.pt \
    data=dataset/license_plate.yaml \
    epochs=50 imgsz=640 batch=8 \
    device=mps
