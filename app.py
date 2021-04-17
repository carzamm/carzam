# 
# app.py
#
# Created On: 4/17/2020
# Last Updated On: 4/17/2020
# Description: Application for "Carzam" web app, a 
#			   Computer Vision Car Identifier
# 

#
# Dependencies
#

# Reference: 
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename   
from flask import send_from_directory
from werkzeug.middleware.shared_data import SharedDataMiddleware

PATH = os.getcwd()
UPLOAD_FOLDER = os.path.join(PATH, 'static/uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
            flash('No file part')
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

			# Load Trained resnet50 Model
            # Using resnet50 as Heroko's free tier size is 512mb, and resnet50
            # is on the smaller size for pretrained models while still offering
            # performance
      		#       model_transfer = models.resnet50(pretrained=False)
		    # model_transfer.classifier=nn.Sequential(nn.Linear(1024,512),
		    #                                     nn.ReLU(),
		    #                                     nn.Dropout(0.2),
		    #                                    nn.Linear(512,133))
		    # model_transfer.load_state_dict(torch.load('model_transfer.pt',map_location='cpu'))

            return render_template('index.html', filename = filename)

    # return redirect(request.url)
    return render_template('index.html')

# Route To Serve Image
# Reference: 
# https://roytuts.com/upload-and-display-image-using-python-flask/
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename = 'uploads/' + filename), code = 307)

# # Prediction Route
# @app.route('/predict',methods=['POST'])
# def predict():

# run app.py on local host port 8080
if __name__ == "__main__":               
    app.run(host='127.0.0.1',port = 8080) 