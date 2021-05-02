rm -r ./ai-classifier/output
python3 split_car_data.py ./ai-classifier/input ./ai-classifier/split 70 0 30
mkdir -p .ai-classifier/output/verify
python3 _dataset_cropper.py
cp -r ./ai-classifier/split/verify ./ai-classifier/output/verify
rm -r ./ai-classifier/split