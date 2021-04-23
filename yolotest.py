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

# Images
indir = './data/testimages/'
outdir = './data/outimages/'
imgs = [indir + f for f in os.listdir(indir)]  # batched list of images

crop_instructions = [] # list of 5-tuples (filename, x-left, y-top, x-right, y-bottom)

for img in imgs:

    # Inference
    results = model(img).tolist()

    # Clean up the results by converting from Tensor to List
    for result in results:

        # Bring the 'xyxy' tensors off the GPU back to the CPU and convert to a list
        location = result.xyxy.cpu().numpy().tolist()

        # Set some variables to track the pixel coordinates
        x_left = int(location[0][0])
        y_top = int(location[0][1])
        x_right = int(location[0][2])
        y_bottom = int(location[0][3])

        # Output the finding to the console
        print("\nFile ---> {}".format(img))
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