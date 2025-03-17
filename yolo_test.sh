yolo \
    task=detect \
    mode=predict \
    model=runs/detect/train/weights/best.pt \
    source=dataset/images/val \
    device=mps
