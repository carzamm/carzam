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
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
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
# Create And Configure Flask Instance
#     

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Max File Size Before Upload Is Aborted
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Disable Cache To Display New Image On Overwrite
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 
# Helper Functions
#

# Confirm extension is valid
def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS          

# Saves the uploaded file to the upload_folder and returns a string
# representative of the path to the file
def save_file_to_upload_directory(file: FileStorage):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

#
# Routing
#

# Home Route
@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # This looks like the case when the file was uploaded and it was valid
        if file and allowed_file(file.filename):
            path_to_file = save_file_to_upload_directory(file)
            crop_instructions = recognize_objects(path_to_file)
            cropped_file_list = generate_cropped_images(OUT_DIRECTORY, crop_instructions)
            print(file.filename)
            print(cropped_file_list)
            # sending a list of files to index.html
            # need a list to send the unformatted result in the event
            # no cropping was done
            list_for_image = []
            # in the event that the list is empty
            if not cropped_file_list:
                print("there are no pictures in the list")
                file_to_get = IN_DIRECTORY + file.filename
                print(IN_DIRECTORY + file.filename)
                list_for_image.append(file_to_get)
                return render_template('index.html', filename = list_for_image)
            else:
                # all the images are placed in a table and shown to user
                return render_template('index.html', filename = cropped_file_list)
            #return str(cropped_file_list)

    # return redirect(request.url)
    return render_template('index.html')

# Route To Serve Image
# changed this route
@app.route('/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename = filename), code = 307)

# run app.py on local host port 8080
if __name__ == "__main__":               
    app.run(host='127.0.0.1',port = 8080) 