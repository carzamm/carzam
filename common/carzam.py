"""
Project: Carzam - CS 467 Capstone
Filename: carzam.py
Description: The main file for carzam. Creates an observer that watches an input directory,
and then waits for new files to be added, processing them as they are
"""

# Reference:
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/

# Dependencies we added from outside sources
import os
from pathlib import Path

# Our own dependencies
from cropper import generate_cropped_images
from recognizer import Recognizer
from identify import Identifier

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# CONSTANTS - CAN BE CHANGED
OUT_DIRECTORY = "./static/outimages/"
IN_DIRECTORY = "./static/testimages/"
INSTRUCTIONS_IMG = "./static/ui-images/instructions.svg"

# Make the directories declared as constants if they do not exist
Path(IN_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(OUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
PATH = os.getcwd()
UPLOAD_FOLDER = os.path.join(PATH, IN_DIRECTORY[2:])

recognizer = Recognizer()
identifier = Identifier(weights_and_biases="./common/deployed.pt", web=True)

#
# Helper Functions
#

# Confirm extension is valid
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if '.' in filename and \
		ext in ALLOWED_EXTENSIONS:
        return True
    else:
        return [(INSTRUCTIONS_IMG, "." + ext + " not supported", None)]       

def parse_file(file):
    print(f'file: {file}')

    crop_instructions = recognizer.recognize_objects(file)
    print(f'crop_instructions: {crop_instructions}')

    # If there is no vehicle in the image, return error message
    if not crop_instructions:
        return [(INSTRUCTIONS_IMG, "No vehicle detected", None)]


    # Now returns (bool, cropped_file_list) 2-tuple
    # Bool is false if it didn't crop any files (i.e. they were all too small)
    cropped_file_list = generate_cropped_images(OUT_DIRECTORY, crop_instructions)[1]
    print(f'cropped_file_list: {cropped_file_list}')

    if not cropped_file_list:
        return [(INSTRUCTIONS_IMG, "Vehicle in image too small", None)]

    # Run the cropped_file_list through the AI Identifer
    results = identifier.test_all_cars(list_of_paths=cropped_file_list)
    print(f'results: {results}')

    # If the confidence of the prediction is below 6.7
    if results[0][2] < 6.7:
        return [(results[0][0], "Vehicle not recognized", None)]

    # in the event that the list is empty
    if not cropped_file_list:
        print("there are no pictures in the list")
        print(file)
        file_to_get = os.path.basename(os.path.normpath(file))
        print(IN_DIRECTORY + file_to_get)
        cropped_file_list.append(IN_DIRECTORY + file_to_get)
    return results

