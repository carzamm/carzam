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
- [x] 1. Install python 3 in docker image [x]
- [x] 2. Install pip3 in docker image
- [x] 3. Copy app source code in docker image
- [x] 4. Install dependencies in docker image
- [x] 5. Expose port in docker image
- [x] 6. Create the docker image
- [x] 7. Run the docker image to create a container instance

#### Docker CLI Commands:

Build the docker image:
> docker build -t flaskapp:latest .

> docker build -t <new image name>:<image version> .
> 1. docker: calls docker client
> 2. builder: used to create a new docker image
> 3. -t: flag that defines the name of the image we are going to create
> dot notation at the end: docker will look for the Dockerfile in the current folder

Show docker images present on local system (after creating some):
> docker images

Run a docker image in demo mode and bind local host to the exposed docker port:
> docker run -it -d -p 8080:8080 flaskapp

> docker run -it -d -p <local port>:<exposed container port> <image name>
> 1. -d: demo flag
> 2. -p: publish flag, publishing containers exposed port to host, allowing host port to bind to container port
> 3. <local port>: this can be any port number you want for local host
> 4. <exposed container port>: not sure this is required with dynamic host allocation, will have to test

Check what containers are currently running on your system:
> docker ps

Stop a container running as an executable (running in perpetuity):
> docker stop <container id>

Exit a container not running as an executable:
> exit

See all docker container's (stopped and running):
> docker ps -a

#### **Docker CLI Clearnup Commands:**

Clean up images/containers and other resources that are dangling:
> docker system prune

Delete a specific image:
> docker rmi <image id>

Delete multiple images:
> docker rmi <image id> <image id>
