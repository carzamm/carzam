import sys
from distutils.dir_util import copy_tree
import os
from os import path
import shutil
import time

def mergeSets(destDir):

    # get a list of each class directory and iterate over it
    dirList = os.listdir(destDir + '/train')
    for directory in sorted(dirList, key=lambda s: s.lower()):

        # ignore hidden files
        if not directory.startswith('.'):

            # ensure the class directory exists in both the source and destination
            if path.exists(destDir + "/train/" + directory) and path.exists(destDir + "/test/" + directory):

                # get list of photos in the class directory and iterate over them
                files = os.listdir(destDir + "/test/" + directory)
                for f in sorted(files):

                    # abitrary iterator to prevent overwriting images
                    i = 0

                    # ignore hidden files
                    if not f.startswith('.'):

                        # ensure the image moving wont overwrite another image
                        if not path.exists(destDir + "/train/" + directory + "/" + f):
                            shutil.move(destDir + "/test/" + directory + "/" + f, destDir + "/train/" + directory)
                        else:
                            # if the file name is duplicated, rename the file being moved
                            shutil.move(destDir + "/test/" + directory + "/" + f, destDir + "/train/" + directory + "/" + str(i) + f)
                        i += 1
            else:
                print(f'ERROR: Directory "{directory}" missing in either the source or destination directory')
                exit()



    return 0

def splitSet(destDir, valInt, testInt):

    # get a list of each class directory and iterate over it
    dirList = os.listdir(destDir + '/train')
    for directory in sorted(dirList, key=lambda s: s.lower()):
        print(directory)
        # ignore hidden files
        if not directory.startswith('.'):

            # ensure the class directory exists in both the source and destination
            if path.exists(destDir + "/train/" + directory) \
                    and path.exists(destDir + "/val/" + directory)\
                    and path.exists(destDir + "/test/" + directory):

                # get list of photos
                files = os.listdir(destDir + "/train/" + directory)
                numFiles = len(files)

                # Round float to nearest int
                numVal = round(numFiles * (valInt / 100))
                numTest = round(numFiles * (testInt / 100))

                # iterate each photo in the class directory
                i = 0



                print(sorted(files))
                for f in sorted(files):
                    print(f)

                    # ignore hidden files
                    if not f.startswith('.'):

                        # Move the appropriate amount of files into the val and test directories
                        if i < numVal:
                            shutil.move(destDir + "/train/" + directory + "/" + f, destDir + "/val/" + directory)
                            files_in_train = i
                        elif i >= numVal and i < (numVal + numTest):
                            shutil.move(destDir + "/train/" + directory + "/" + f, destDir + "/test/" + directory)
                    i += 1

def create_folders(sourceDir, destDir):
    # Get the car names and create that folder structure
    car_folders = [directory for directory in os.listdir(sourceDir) if os.path.isdir(os.path.join(sourceDir, directory))]

    # create the output folders
    dest_test = os.path.join(destDir, "test")
    dest_val = os.path.join(destDir, "val")

    for top_level_folder in [dest_test, dest_val]:
        os.makedirs(top_level_folder)
        for car in car_folders:
            os.makedirs(os.path.join(top_level_folder, car))

def main():
    if len(sys.argv) != 6 or int(sys.argv[3]) + int(sys.argv[4]) + int(sys.argv[5]) != 100:
        print(f'### USAGE ###')
        print(f'python3 split_car_data.py source_directory destination_directory train_int validation_int test_int')
        print(f'e.g. "python3 split_car_data.py car_data 70 15 15')
        print('Note: train, validation, and test must add up to 100')
        return 0

    sourceDir = sys.argv[1]
    destDir = sys.argv[2]
    valInt = int(sys.argv[4])
    testInt = int(sys.argv[5])
    


    # copy data
    copy_tree(sourceDir, os.path.join(destDir, "train"))
    
    # create folders
    create_folders(sourceDir, destDir)

    mergeSets(destDir)

    # create val directory
    copy_tree(destDir + "/test", destDir + "/val")

    splitSet(destDir, valInt, testInt)


    return 0

if __name__ == "__main__":
    main()