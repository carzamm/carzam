# Carzam
The CS467 Capstone Project at Oregon State University in the Spring of 2021. Carzam is a web application that leverages an AI to identify vehicles, their make and model in a submitted picture.

 ### To test:
 Create and setup a virtual environment:
 > python3 -m venv venv  
 > source ./venv/bin/activate  

 Install dependencies:  
 > pip3 install -r requirements.txt

 Run locally:  
 > python3 app.py

## Docker Reference:

### Getting set up:

Dowload and install Docker:
> https://docs.docker.com/get-docker/


Ensure the Docker cli is working:
> docker --version


Basic Docker workflow (handled in the Dockerfile):
> 1. Install python 3 in Docker image
> 2. Install pip3 in Docker image
> 3. Copy app source code in Docker image
> 4. Install dependencies in Docker image
> 5. Expose port in Docker image
> 6. Create the Docker image
> 7. Run the Docker image to create a container instance

### Workflow for testing locally:
> Testing locally without docker still works with "python3 app.y", you just want to make sure you test the Docker build and run steps when you are ready to make a pull request.
>
> When ready for a pull request:
> 1. Build the Docker image
> 2. Run the Docker image to create a container
> 3. View app on local host to confirm Docker container is functional
> 4. Use the Docker desktop app to view Docker server logs if an issue arises
> 5. Occassionally clean up dangling Docker images from old deployments

### Docker CLI Commands:

Build the docker image:
> docker build -t flaskapp:latest .

> docker build -t <new image name>:<image version> .
> 1. docker: calls Docker client
> 2. builder: used to create a new Docker image
> 3. -t: flag that defines the name of the image we are going to create
> dot notation at the end: Docker will look for the Dockerfile in the current folder

Run a docker image in demo mode and bind local host to the exposed docker port:
> docker run -it -d -p 8080:8080 flaskapp

> docker run -it -d -p <local port>:<exposed container port> <image name>
> 1. -d: demo flag
> 2. -p: publish flag, publishing containers exposed port to host, allowing host port to bind to container port
> 3. <local port>: this can be any port number you want for local host
> 4. <exposed container port>: not sure this is required with dynamic host allocation, will have to test

Show docker images present on local system (after creating some):
> docker images

Check what containers are currently running on your system:
> docker ps

Stop a container running as an executable (running in perpetuity):
> docker stop <container id>

Exit a container not running as an executable (running in the foreground of shell):
> exit

See all container's (stopped and running):
> docker ps -a

### Docker CLI Clearnup Commands:

Clean up images/containers and other resources that are dangling:
> docker system prune

Delete a specific image:
> docker rmi <image id>

Delete multiple images:
> docker rmi <image id> <image id>
