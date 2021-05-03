# MUST ADD UP TO 100
TRAIN=75
TEST=25
VALIDATE=0

# Creates necessary directories, runs them through Kevin's splitter
# and then runs them through the cropper, everything neatly
# placed in /output and excess cleaned up.
rm -r ./ai-classifier/output
python3 split_car_data.py ./ai-classifier/input ./ai-classifier/split $TRAIN $VALIDATE $TEST
mkdir -p .ai-classifier/output/verify
python3 dataset_cropper.py
cp -r ./ai-classifier/split/verify ./ai-classifier/output/verify
rm -r ./ai-classifier/split
python3 count.py