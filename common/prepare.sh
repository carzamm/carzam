# MUST ADD UP TO 100
TRAIN=75
TEST=25
VALIDATE=0

# DATASET TO USE
DATASET="carzam"

# Creates necessary directories, runs them through Kevin's splitter
# and then runs them through the cropper, everything neatly
# placed in /output and excess cleaned up.
rm -r ../data/input
python3 dataset_cropper.py ../data carzam
python3 split_car_data.py "../data/${DATASET}" ../data/input $TRAIN $VALIDATE $TEST
mkdir -p ../data/input/verify
python3 count.py