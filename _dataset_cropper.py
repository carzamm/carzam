from cropper import generate_cropped_images
from recognizer import recognize_objects
import os

INPUT_DIR = "./ai-classifier/input"
OUTPUT_DIR = "./ai-classifier/output"
TARGET_DIRS = ['test', 'train']

for basedir in os.listdir(INPUT_DIR):
    if basedir in TARGET_DIRS:
        print(basedir)
        for subdir in os.listdir(os.path.join(INPUT_DIR, basedir)):
            print("\t{}".format(subdir))
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(INPUT_DIR, basedir, subdir)):
                for file in filenames:
                    print("\t\t{}".format(file))
                    print("Processing File...")
                    image_loc = os.path.join(INPUT_DIR, basedir, subdir, file)
                    print(image_loc)
                    if os.path.isfile(image_loc):
                        crop_instructions = recognize_objects(image_loc)[:1]
                        print(crop_instructions)
                        if not os.path.exists(os.path.join(OUTPUT_DIR, basedir, subdir)):
                            os.makedirs(os.path.join(OUTPUT_DIR, basedir, subdir))
                            # generate_cropped_images(os.path.join(OUTPUT_DIR, basedir, subdir), crop_instructions)
                    print("Done Processing File!")