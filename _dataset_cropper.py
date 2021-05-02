"""
Project: Carzam - CS 467 Capstone
Filename: _dataset_cropper.py (Utility)
Description: Scans the input directory /test and /train and crops all photos
in those directories creating copies with the same folder structure.

For complicated photonames, filenames are SHA256 hashes of the original filename.
"""

from cropper import generate_cropped_images
from recognizer import recognize_objects
import os

INPUT_DIR = "./ai-classifier/input"
OUTPUT_DIR = "./ai-classifier/output"
TARGET_DIRS = ['test', 'train']

def get_largest(crop_instructions):
    print(crop_instructions)
    maxheight, maxwidth = 0, 0
    best = None
    for instruction in crop_instructions:
        _, _, leftx, topy, rightx, bottomy = instruction
        height = bottomy - topy
        width = rightx - leftx
        if height > maxheight and width > maxwidth:
            maxheight = height
            maxwidth = width
            best = instruction
    return [best]
        

for basedir in os.listdir(INPUT_DIR):
    if basedir in TARGET_DIRS:
        print(basedir)
        for subdir in os.listdir(os.path.join(INPUT_DIR, basedir)):
            print("\t{}".format(subdir))
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(INPUT_DIR, basedir, subdir)):
                for file in filenames:
                    print("\t\t{}".format(file))
                    print("Processing File...")
                    image_loc = os.path.join(INPUT_DIR, basedir, subdir, file)
                    print(image_loc)
                    if os.path.isfile(image_loc):
                        crop_instructions = recognize_objects(image_loc)
                        crop_instructions = get_largest(crop_instructions)
                        print(crop_instructions)
                        if not os.path.exists(os.path.join(OUTPUT_DIR, basedir, subdir)):
                            os.makedirs(os.path.join(OUTPUT_DIR, basedir, subdir))
                        generate_cropped_images(os.path.join(OUTPUT_DIR, basedir, subdir), crop_instructions, min_size=(0, 0))
                    else:
                        print("NOT A FILE: {}".format(image_loc))
                    print("Done Processing File!")