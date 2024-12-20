from ultralytics import YOLO
import torch
import os

# Check for MPS availability
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
model = YOLO("yolo11x.pt")

# Adjust num_workers for your system
num_workers = min(os.cpu_count(), 4)  # Limit workers to avoid overload

# Train the model
model.train(
    data="dataset_gesture.yaml", 
    imgsz=640, 
    batch=8, 
    epochs=20,
    workers=num_workers,
    verbose=True
)
