"""
yolotest.py
"""
import torch
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
dir = './'
imgs = [dir + f for f in ('cars.jpg',)]  # batched list of images

# Inference
results = model(imgs)

# Results
# results.print()  
# results.save()  # or .show()

# Data Processing Section

# Extract the coordinates of the bounding box from the YOLOv5 results
rough_coordinates = list(results.xyxy[0].cpu().numpy())

# Create a list of coordinate lists we can pass to the Photo Cropper
coordinate_list = []
for row in rough_coordinates:
    row = list(row)
    row = row[:4]
    coordinate_list.append(row)

print(coordinate_list)