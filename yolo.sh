#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 [train|predict]"
  exit 1
fi

MODE=$1

if [ "$MODE" == "train" ]; then
  echo "Running YOLO training..."
  yolo \
    task=detect \
    mode=train \
    model=yolov8n.pt \
    data=dataset/license_plate.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    device=mps
elif [ "$MODE" == "predict" ]; then
  echo "Running YOLO prediction..."
  yolo \
    task=detect \
    mode=predict \
    model=runs/detect/train/weights/best.pt \
    source=dataset/images/val \
    device=mps
else
  echo "Invalid mode: $MODE. Use 'train' or 'predict'."
  exit 1
fi
