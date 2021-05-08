# define base docker image for the app
# we are choosing python base image, version 3.7-slim
# python:3.7-slim speeds up dependency installation
from python:3.7-slim

# create working directory inside the docker image
WORKDIR /srv

# https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project
ADD ./requirements.txt /srv/requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt
ADD . /srv
# RUN python setup.py install


# copy carzam app into docker image
# dot notation says source for copy is all files in current directory
# current directory is the external directory Dockerfile is located
# destination is the internal docker image directory "/app"
# COPY . .

# install dependencies in the docker image
# it is installing from the interal docker image "/app" directory
# add --no-chache-dir to keep docker image size smaller, build fails with out it
# RUN pip3 install -r requirements.txt

# solves "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
#https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# required by heroku documentation
# exposes a dynamic port to the outside world 
# port is determined by heroku
CMD gunicorn --bind 0.0.0.0:$PORT wsgi

# make containers created from this docker image executable
# ENTRYPOINT makes the container executable
# [ ] holds commands we want to run when creating container
# CMD [ ] containers arguments to pass to ENTRYPOINT command 
# CMD can pass multiple arguments... [ "<arg1>", "<arg2>"]
ENTRYPOINT ["python3"]
CMD ["app.py"]
