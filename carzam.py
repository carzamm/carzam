"""
Project: Carzam - CS 467 Capstone
Filename: carzam.py
Description: The main file for carzam. Creates an observer that watches an input directory,
and then waits for new files to be added, processing them as they are
"""

# Reference: 
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename   
from flask import send_from_directory
from werkzeug.middleware.shared_data import SharedDataMiddleware
from pathlib import Path
from observer import directory_observer

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# CONSTANTS - CAN BE CHANGED
OUT_DIRECTORY = "./static/outimages/"
IN_DIRECTORY = "./static/testimages/"

# Make the directories declared as constants if they do not exist
Path(IN_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(OUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
PATH = os.getcwd()
UPLOAD_FOLDER = os.path.join(PATH, IN_DIRECTORY[2:])

# Create the observer to scan IN_DIRECTORY and then output to OUT_DIRECTORY
# observer = directory_observer(IN_DIRECTORY, OUT_DIRECTORY)
# observer.start()
# observer.join()

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
        if file and allowed_file(file.filename):
            filename = secure_filename("image.png")

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template('index.html', filename = filename)

    # return redirect(request.url)
    return render_template('index.html')

# Route To Serve Image
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename = 'testimages/' + filename), code = 307)

# run app.py on local host port 8080
if __name__ == "__main__":               
    app.run(host='127.0.0.1',port = 8080) 