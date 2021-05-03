#!/bin/bash

# Project: Carzam
# Filename: prepare.sh
# Description: Shell script to take a regular dataset located in /data and change it from
#   /data/{car}/file.jpg format into a balanced dataset in /data/[test,train,verify]/{car}/file.jpg
#   format. Crops all the cars, expelling anything under 400x400 and then sorts with Kevin's script.
# 
# TO RUN:   prepare.sh [dir in data with dataset] [outdir in data] [train] [test] [verify]
# Example:  prepare.sh CARZAM ready 70 30 0

if [ $# -lt 5 ]
then
    printf "Insufficent arguments provided\nTO RUN: prepare.sh [dir in data with dataset] [outdir in data] [train] [test] [verify]\n"
    exit 1
fi

# MUST ADD UP TO 100
TRAIN=$3
TEST=$4
VERIFY=$5

# DATASET TO USE
DATASET=$1
OUTPUT_DIR=$2


# Delete the old directories
if [ -d "../data/cropped" ]
then
    rm -r "../data/cropped"
fi 

if [ -d "../data/${OUTPUT_DIR}" ]
then
    rm -r "../data/${OUTPUT_DIR}"
fi  

# Crop the photos from desired dataset
python3 dataset_cropper.py $DATASET

# Use Kevin's script to balance the data
python3 split_car_data.py ../data/cropped "../data/${OUTPUT_DIR}" $TRAIN $VERIFY $TEST

# Delete the new 'cropped' directory
if [ -d "../data/cropped" ]
then
    rm -r "../data/cropped"
fi  

python3 count.py "../data/${OUTPUT_DIR}"