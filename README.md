# carzam
The CS467 Capstone Project at Oregon State University in the Spring of 2021. Carzam is a web application that leverages an AI to identify vehicles, their make and model in a submitted picture.

 #### To test:
 Ideally, you should setup a virtual environment before running this:
 > python3 -m venv venv  
 > source ./venv/bin/activate  

 Install dependencies by doing:  
 > pip3 install -r requirements.txt

 Then run by doing:  
 > python3 app.py

## Docker Notes

#### Getting set up:

Dowload and install Docker:
> https://docs.docker.com/get-docker/

Ensure the docker cli is working:
> docker --version

Basic Docker workflow (handled by Dockerfile):
> 1. Install python 3 in docker image
> 2. Install pip3 in docker image
> 3. Copy app source code in docker image
> 4. Install dependencies in docker image
> 5. Expose port in docker image
> 6. Create the docker image
> 7. Run the docker image to create a container instance

#### Docker CLI Notes:
