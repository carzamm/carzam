# Carzam

The CS467 Capstone Project at Oregon State University in the Spring of 2021. Carzam is a web application that leverages an AI to identify vehicles, their make and model in a submitted picture.

## Running The Project
Create and setup a virtual environment:
```bash
python3 -m venv venv  
source ./venv/bin/activate
```

Install dependencies:  
```bash
pip3 install -r requirements.txt
```

Run locally:  
```bash
python3 app.py
```

## Project Structure

A brief overview of project structure. This project is getting huge quick, so I tried to do a bit of housekeeping to keep everything tidy.

|File|Description|
|----|-----------|
|`app.py`| Entry point for the web server |
|`requirements.txt`| Up-to-date list of dependencies for the project. If you install new dependencies be sure to update by running `pip3 freeze > requirements.txt` in the root of the project. Make sure you are in your virtual envirionment otherwise you'll add all the dependencies from your computer's Python installation.|
|`.gitignore`| Self explanatory. Add what you need.|
|`/data`| Dataset directory. Not saved to this repo, because its MASSIVE. Just follow the notes in the **AI Identifier** section. |
|`/common`| All Python scripts go here. Keeps the root of the project from getting cluttered. |
|`Dockerfile`| Kevin can probably explain this one better |
|`/static`| Used by the web interface to serve static assets like JS, CSS or images.|
|`/project-progress`| Folder to store screenshots of interesting things you encounter during project development, or big milestones.|
|`/templates` | Used by the web interface for dynamic content |

Inside of `common`, what does what...

|File|Description|
|----|-----------|
|`carzam.py`| Supporting functions for `app.py` |
|`count.py` | Script called by `prepare.sh` to count the files that were converted during the preparation process.|
|`cropper.py`| The cropping module. Takes `crop_instructions` and saves cropped images.|
|`dataset_cropper.py`| Used by `prepare.sh` to crop all the files. Helper module.|
|`identify.py`| The vehicle identification modules. Takes a trained model and guesses what car is in a presented image.|
|`recognizer.py`| Object recognition module. Identifies cars and trucks in an image. Generates `crop_instructions`.|
|`split_car_data.py`| Kevin's balancing script. Sorts images from one directory into `test`, `train` and `verify` subdirectories based on user arguments.|
|`train.py`| Utility module to train the vehicle identification AI. |

And the current models (held in `common`)...

|File|Description|
|----|-----------|
|`yolov5.pt`| Not in repo, but if you see it, `recognizer.py` relies on this.|
|`deployed.pt`| A somewhat decent model at guessing from the 9 car classes in the CARZAM dataset.|
|`saved_model.pt`| The output from `train.py` and might be great, might not be. As of the time of writing this, it only had a 47% accuracy rate.|



# AI Identifier

This is the component that identifies the make and model of a vehicle in an image.

A couple of **IMPORTANT** notes.

- All datasets are now going to be manipulated/stored/kept in `/data`. The exception to this is the place that files are held when they are uploaded, for now, that's not a big deal.

- The convention of directory names within `/data` is:

-   - `UPPERCASE` for original datasets that have not been cleaned yet. They should be a folder, that contains folders with car names, and car pictures within each of those car named folders.

-   - `lowercase` for datasets that have been manipulated, such as what comes out of the `prepare.sh` script. These datasets should be a folder with a lowercase name, containing three folders, namely `test`, `train` and `verify`. I changed `validate` to `verify` everywhere because it was shorter. We can use the term `validate` to mean any image we are testing a completed model against (i.e. a user-uploaded image). Within the three folders, there should be folders with car names each containing 'cleaned' images of the cars.

## Training

1. Have a dataset that looks like this:

```bash
dataset
┣ Car 1
┃  ┣ anyname.jpg
┃  ┣ anyname.png
┃  ┣ anyname2.jpg
┃  ┗ anyname2.jpg
┃ 
┣ Car 2
┣ Car 3
┣ Car 4
┣ Car 5
┗ Car 6
```

2. Place it in the `data` directory of the root of this project.

3. Navigate to `common`

4. Run: 

```bash
prepare.sh [your dateset folder name] [desired output folder name] [test %] [train %] [verify %]

# Example: prepare.sh CARZAM ready 70 30 0
```

5. At this point, whatever files in the dataset will be cropped, the ones that are smaller than 400x400 are discarded, and then Kevin's splitter will separate them into test/train/verify.

6. You can find your 'clean' dataset in `/data/[output]` where output is whatever you type for desired output folder name when you ran `prepare.sh`

7. Still working in `common` run:

```bash
python3 train.py [your clean dataset name]

# Example: python3 train.py ready
```

8. The AI will use your dataset to train 20 iterations, you can change these settings inside of `train.py` if you wish. That's part of the fun. Your saved weights and balances will be written as `saved_model.pt` in the `common` folder.

9. Done! You just trained an AI!


## Testing the Model (Identify Cars from Images in a Folder)

1. Make sure you have a 'clean' dataset in `/data`. If not, go up one paragraph and read 'Training'.

2. Make sure your trained model is named `saved_model.pt` and it is in the `common` directory.

3. Run:

```bash
python3 identify.py [name of directory with validation images in /data]

# Example: python3 identify.py VALIDATION
```

4. Output will be displayed on the screen with the guesses of the model along with their confidence values.

# Docker

The project has been containerized!

## Getting set up:

Dowload and install Docker:
> https://docs.docker.com/get-docker/


Ensure the Docker CLI is working:
> docker --version


Basic Docker workflow (handled in the Dockerfile):
> 1. Install python 3 in Docker image
> 2. Install pip3 in Docker image
> 3. Copy app source code in Docker image
> 4. Install dependencies in Docker image
> 5. Expose port in Docker image
> 6. Create the Docker image
> 7. Run the Docker image to create a container instance

## Workflow for testing locally:
> Testing locally without docker still works with "python3 app.y", you just want to make sure you test the Docker build and run steps when you are ready to make a pull request.
>
> When ready for a pull request:
> 1. Build the Docker image
> 2. Run the Docker image to create a container
> 3. View app on local host to confirm Docker container is functional
> 4. Use the Docker desktop app to view Docker server logs if an issue arises
> 5. Occassionally clean up dangling Docker images from old deployments

## Docker CLI Commands:

Build the docker image:
> docker build -t flaskapp:latest .

> docker build -t 'new image name':'image version' .
> 1. docker: calls Docker client
> 2. builder: used to create a new Docker image
> 3. -t: flag that defines the name of the image we are going to create
> dot notation at the end: Docker will look for the Dockerfile in the current folder

Run a docker image in demo mode:
> docker run -it -d -p 8080:8080 flaskapp

> docker run -it -d -p 'local port':'exposed container port' 'image name'
> 1. -d: demo flag
> 2. -p: publish flag, publishing containers exposed port to host, allowing host port to bind to container port
> 3. <local port>: this can be any port number you want for local host
> 4. <exposed container port>: not sure this is required with dynamic host allocation, will have to test

Show docker images present on local system:
> docker images

Check what containers are currently running on your system:
> docker ps

Stop a container running as an executable (running in perpetuity):
> docker stop 'container id'

Exit a container not running as an executable (running in the foreground of shell):
> exit

See all container's (stopped and running):
> docker ps -a

## Docker CLI Clearnup Commands:

Clean up images/containers and other resources that are dangling:
> docker system prune

Delete a specific image:
> docker rmi 'image id'

Delete multiple images:
> docker rmi 'image id' 'image id'
