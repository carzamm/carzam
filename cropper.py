"""
Project: Carzam - CS 467 Capstone
Filename: cropper.py
Description: Single function that accepts a list of crop_instructions in the format

[(filename, index, x-left, y-top, x-right, y-bottom), ...]

and outputs cropped image files using those coordinates
"""

from PIL import Image


def generate_cropped_images(out_directory: str, crop_instructions: list):
    """
    Main function for cropper.py. Generates the cropped images from an list crop_instructions
    out_directory should be the path as a string to where the output files should be placed
    crop_instructions will be a list of tuples in the format of:

    [(filename, index, x-left, y-top, x-right, y-bottom), ...]

    """

    # Iterate through every tuple in crop_instructions
    for instruction in crop_instructions:

        # Open the file, convert to RGB (handles JPG and PNG)
        main_image = Image.open(instruction[0]).convert("RGB")

        # Save the index number for filename
        index = instruction[1]

        # Save the coordinates
        left = instruction[2]
        top = instruction[3]
        right = instruction[4]
        bottom = instruction[5]

        # Save the filename without the path

        filename = instruction[0].split('/')[-1]
        name, extension = filename.rsplit('.')

        # Crop the image using the given coordinates
        cropped_image = main_image.crop((left, top, right, bottom))
        path_to_save_file = out_directory + name + "_{}.".format(index) + extension

        # Save the new file
        cropped_image.save(path_to_save_file, "JPEG")
