"""
Project: Carzam - CS 467 Capstone
Filename: _dataset_cropper.py (Utility)
Description: Utility - Reads a directory from /data and runs all photos through the 
cropper.

Run: python3 dataset_cropper.py [dataset]
Example: python dataset_cropper.py carzam
                                   *Since carzam is located at /data/carzam

For complicated photonames, filenames are SHA256 hashes of the original filename.
"""

import os
import sys
from cropper import generate_cropped_images
from recognizer import Recognizer
import shutil

def get_largest(instructions):
    """ Gets the largest picture in the crop instructions """

    maxheight, maxwidth = 0, 0
    best = None
    for instruction in instructions:
        _, _, leftx, topy, rightx, bottomy = instruction
        height = bottomy - topy
        width = rightx - leftx
        if height > maxheight and width > maxwidth:
            maxheight = height
            maxwidth = width
            best = instruction
    return [best]


if __name__ == "__main__":

    # Get the variables from the command line
    if (len(sys.argv) < 2):
        print("Not enough arguments.\nUsage >> python3 dataset_cropper.py [dataset]\nWhere [dataset] is the folder in /data")
        exit()
    DATASET = sys.argv[1]
    TARGET_DIRS = ['test', 'train']
    DATASET_DIR = os.path.join("../data", DATASET)
    OUTPUT_FOLDER = '../data/cropped'

    # Start a recognizer object and declare variable to let us know how many images were processed.
    recognizer = Recognizer()
    imgs_processed = 0
    imgs_rejected = 0

    # It expects the dataset to be in the format /dataset_name/[class_of_car]/*
    # Example:  /carzam/Acura TL/* or /carzam/Audi A4/*
    for in_car_dir in os.listdir(DATASET_DIR):
        
        # Save the car name for the current directory, the path to the directory and get
        # the list of files in the directory
        car_name = in_car_dir
        in_car_dir = os.path.join(DATASET_DIR, in_car_dir)
        files = os.listdir(in_car_dir)

        # Create the new folder for the car if it doesn't exist
        out_car_dir = os.path.join(OUTPUT_FOLDER, car_name)
        if not os.path.exists(out_car_dir):
            os.makedirs(out_car_dir)
        
        # Begin cropping each of the files
        for file in files:

            # The exact relative path to the file
            image_loc = os.path.join(in_car_dir, file)

            # Make sure its a valid file (and not a directory)
            if os.path.isfile(image_loc):

                # Identify the cars and trucks in the photograph
                crop_instructions = recognizer.recognize_objects(image_loc)

                # If we got nothing back, report an error
                if len(crop_instructions) == 0:
                    print("File Error: {}".format(image_loc))

                # Pick the biggest image in the picture, it's probably the subject of the image
                crop_instructions = get_largest(crop_instructions)

                # Crop the image
                did_crop, _ = generate_cropped_images(
                    out_car_dir, 
                    crop_instructions, 
                    min_size=(200, 200), 
                    padding=True)

                # Keep track of successes and failures.
                if (did_crop):
                    imgs_processed += 1
                else:
                    imgs_rejected += 1
            
        # check whether the out folder for a certain car file
        # is empty, if it is then it removes from
        # the list of directories
        # (makes splitting easier)
        if len(os.listdir(out_car_dir)) < 5:
            shutil.rmtree(out_car_dir)
            print("Removed directory", out_car_dir, file=sys.stdout)

    # Print a status to the screen when the job is done.
    print("Done!\nCropped: {}\tRejected: {}".format(imgs_processed, imgs_rejected))
