# define base docker image for the app
# we are choosing latest image of alpine for our base image
# alpine is a lightweight linux system
# from alpine:latest
from python:3.7-slim

# add the python3 development package and pip 3 to the docker image
# apk is package manager for alpine linux
# --no-cache flag, we don't want cache increasing image size
# py3-pip adds pip3 to the docker image
# RUN apk add --no-cache python3-dev py3-pip && \ 
#	pip3 install --upgrade pip

# create working directory inside the docker image
WORKDIR /app

# dot notation says source for copy is all files in current directory
# current directory is the external directory Dockerfile is located
# destination is the internal docker image directory "/app"
COPY . /app

# install dependencies in the docker image
# it is installing from the interal docker image "/app" directory
# --no-chache-dir will help keep docker image size smaller
RUN pip3 --no-cache-dir install -r requirements.txt

