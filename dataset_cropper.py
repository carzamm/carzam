"""
Project: Carzam - CS 467 Capstone
Filename: _dataset_cropper.py (Utility)
Description: Scans the input directory /test and /train and crops all photos
in those directories creating copies with the same folder structure.

For complicated photonames, filenames are SHA256 hashes of the original filename.
"""

import os
from cropper import generate_cropped_images
from recognizer import Recognizer


INPUT_DIR = "./ai-classifier/split"
OUTPUT_DIR = "./ai-classifier/output"
TARGET_DIRS = ['test', 'train']

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
    recognizer = Recognizer()
    imgs_processed = 0

    for basedir in os.listdir(INPUT_DIR):
        if basedir in TARGET_DIRS:

            for subdir in os.listdir(os.path.join(INPUT_DIR, basedir)):

                for (dirpath, dirnames, filenames) in os.walk(os.path.join(INPUT_DIR, basedir, subdir)):

                    for file in filenames:
                        image_loc = os.path.join(INPUT_DIR, basedir, subdir, file)

                        if os.path.isfile(image_loc):
                            crop_instructions = recognizer.recognize_objects(image_loc)
                            if len(crop_instructions) == 0:
                                print("File Error: {}".format(image_loc))
                            crop_instructions = get_largest(crop_instructions)

                            if not os.path.exists(os.path.join(OUTPUT_DIR, basedir, subdir)):
                                os.makedirs(os.path.join(OUTPUT_DIR, basedir, subdir))

                            generate_cropped_images(os.path.join(OUTPUT_DIR, basedir, subdir), crop_instructions, min_size=(0, 0), padding=True)
                            imgs_processed += 1
    print("Cropped {} images. Done!".format(imgs_processed))