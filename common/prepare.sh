#!/bin/bash

# MUST ADD UP TO 100
TRAIN=75
TEST=25
VALIDATE=0

# DATASET TO USE
DATASET="CARZAM"
OUTPUT_DIR="ready"


# Delete the old 'cropped' directory
if [ -d "../data/cropped" ]
then
    rm -r "../data/cropped"
fi  

# Crop the photos from desired dataset
python3 dataset_cropper.py $DATASET

# Use Kevin's script to balance the data
python3 split_car_data.py ../data/cropped "../data/${OUTPUT_DIR}" $TRAIN $VALIDATE $TEST

# Delete the new 'cropped' directory
if [ -d "../data/cropped" ]
then
    rm -r "../data/cropped"
fi  

python3 count.py "../data/${OUTPUT_DIR}"