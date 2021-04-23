"""
Project: Carzam - CS 467 Capstone
Filename: yolotest.py
Description: Identifies all photographs in ./data/testimages and uses YOLOv5 to identify the cars and trucks in the 
picture. The coordinates of the bounding boxes are returned by the model, recorded and passed to the Python Image
Library, which crops the original photographs, and saves the cropped vehicles in new files located in ./data/testimages
"""
import torch
import numpy as np
from PIL import Image
import os


#
#   IMAGE DETECTOR SECTION ----------------------------------------------------------
#

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set YOLO to only detect cars (id = 2) and trucks (id = 7)
model.classes = [2, 7]

# Images
indir = './data/testimages/'
outdir = './data/outimages/'
imgs = [indir + f for f in os.listdir(indir)]  # batched list of images

crop_instructions = [] # list of 6-tuples (filename, index, x-left, y-top, x-right, y-bottom)

for img in imgs:

    # Inference
    # results will be a list of Detection objects (cars, trucks, etc.)
    results = model(img).tolist()

    # For each detection object, go through all the objects detected for that class of detection
    for detection in results:
        
        # Print the relevant filename
        print("\nFile ---> {}".format(img))

        # Bring the 'xyxy' tensors off the GPU back to the CPU and convert to a list
        vehicles = detection.xyxy.cpu().numpy().tolist()

        # For vehicle detected in the detection object, get the relevant information
        for index, vehicle in enumerate(vehicles):

            # Set some variables to track the pixel coordinates
            x_left = int(vehicle[0])
            y_top = int(vehicle[1])
            x_right = int(vehicle[2])
            y_bottom = int(vehicle[3])

            # Output the finding to the console
            print("\n #{}".format(index + 1))
            print("\tTop-Left (x,y): {},{}".format(x_left, y_top))
            print("\tBottom-Right (x,y): {}, {}".format(x_right, y_bottom))

            # Add this to the crop instructions so the image cropper can generate the new pictures
            crop_instructions.append((str(img), index, x_left, y_top, x_right, y_bottom))

#
#   IMAGE CROPPER SECTION ---------------------------------------------------------------------
#


# Iterate through every tuple in crop_instructions
for instruction in crop_instructions:
    
    # Open the file, convert to RGB (handles JPG and PNG)
    main_image = Image.open(instruction[0]).convert("RGB")

    # Save the index number for filename
    index = instruction[1]

    # Save the coordinates
    left = instruction[2]
    top = instruction[3]
    right = instruction[4]
    bottom = instruction[5]
    
    # Save the filename without the path
    filename = instruction[0].split('/')[-1]
    name, extension = filename.rsplit('.')

    # Crop the image using the given coordinates
    cropped_image = main_image.crop((left, top, right, bottom))

    # Save the new file
    cropped_image.save(outdir + name + "_{}.".format(index) + extension, "JPEG")