"""
yolotest.py
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

crop_instructions = [] # list of 5-tuples (filename, x-left, y-top, x-right, y-bottom)

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
            crop_instructions.append((str(img), x_left, y_top, x_right, y_bottom))

#
#   IMAGE CROPPER SECTION ---------------------------------------------------------------------
#

for instruction in crop_instructions:
    
    main_image = Image.open(instruction[0]).convert("RGB")

    left = instruction[1]
    top = instruction[2]
    right = instruction[3]
    bottom = instruction[4]
    
    filename = instruction[0].split('/')[-1]

    cropped_image = main_image.crop((left, top, right, bottom))
    cropped_image.save(outdir + filename, "JPEG")