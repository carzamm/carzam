"""
yolotest.py
"""
import torch
import numpy as np
from PIL import Image
import os

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
indir = './data/testimages/'
outdir = './data/outimages/'
imgs = [indir + f for f in os.listdir(indir)]  # batched list of images
filenames = [fn for fn in os.listdir(indir)]

image_locations = imgs.copy() # since imgs gets overwritten later, we need to copy the file paths for cropping

# Inference
results = model(imgs)

# Extract the coordinates of the bounding box from the YOLOv5 results
rough_coordinates = list(results.xyxy[2].cpu().numpy())

# Create a list of coordinate lists we can pass to the Photo Cropper
coordinate_list = []
for row in rough_coordinates:
    row = list(row)
    row = row[:4]
    coordinate_list.append(row)

# Crop each image



main_image = Image.open(image_locations[2]).convert("RGB")

left = coordinate_list[0][0]
top = coordinate_list[0][1]
right = coordinate_list[0][2]
bottom = coordinate_list[0][3]

cropped_image = main_image.crop((left, top, right, bottom))
cropped_image.save(outdir + filenames[2], "JPEG")