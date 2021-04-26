"""
Project: Carzam - CS 467 Capstone
Filename: carzam.py
Description: The main file for carzam. Creates an observer that watches an input directory,
and then waits for new files to be added, processing them as they are
"""

# Reference: 
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
import os
import time
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from werkzeug.middleware.shared_data import SharedDataMiddleware
from pathlib import Path
from cropper import generate_cropped_images
from recognizer import recognize_objects

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# CONSTANTS - CAN BE CHANGED
OUT_DIRECTORY = "./static/outimages/"
IN_DIRECTORY = "./static/testimages/"

# Make the directories declared as constants if they do not exist
Path(IN_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(OUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
PATH = os.getcwd()
UPLOAD_FOLDER = os.path.join(PATH, IN_DIRECTORY[2:])

# 
# Helper Functions
#

# Confirm extension is valid
def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS          

def parse_file(file):
    crop_instructions = recognize_objects(file)
    cropped_file_list = generate_cropped_images(OUT_DIRECTORY, crop_instructions)
    print(cropped_file_list)
    # in the event that the list is empty
    if not cropped_file_list:
        print("there are no pictures in the list")
        print(file)
        file_to_get = os.path.basename(os.path.normpath(file))
        print(IN_DIRECTORY + file_to_get)
        cropped_file_list.append(IN_DIRECTORY + file_to_get)
    return cropped_file_list

