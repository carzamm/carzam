"""
Project: Carzam - CS 467 Capstone
Filename: recognizer.py
Description: Uses the YOLOv5 library to detect car and truck objects in a photograph.
"""

import torch


def recognize_objects(path_to_image: str):
    """
    recognize_objects takes one argument, the path to the image it will be recognizing
    objects in. It will run the image through the YOLOv5 AI and will return a list of
    tuples in the format of:

    [(filename, index, x-left, y-top, x-right, y-bottom), ...]

    This datatype is referred to as crop_instructions
    """

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)

    # Set YOLO to only detect cars (id = 2) and trucks (id = 7)
    model.classes = [2, 7]

    # This will be list of 6-tuples (filename, index, x-left, y-top, x-right, y-bottom)
    crop_instructions = []

    # Inference
    # results will be a list of Detection objects (cars, trucks, etc.)
    results = model(path_to_image).tolist()

    # For each detection object, go through all the objects detected for that class of detection
    for detection in results:

        # Print the relevant filename
        print("\nFile ---> {}".format(path_to_image))

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
            crop_instructions.append((str(path_to_image), index, x_left, y_top, x_right, y_bottom))

    return crop_instructions
