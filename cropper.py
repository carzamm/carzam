"""
Project: Carzam - CS 467 Capstone
Filename: cropper.py
Description: Single function that accepts a list of crop_instructions in the format

[(filename, index, x-left, y-top, x-right, y-bottom), ...]

and outputs cropped image files using those coordinates

MIN_SIZE constant is defined to filter out very small images. It's a tuple of 
width in pixels by height in pixels. Images that are smaller than this on EITHER
side will not be processed.
"""

from PIL import Image

MIN_SIZE = 200, 200 # This can be modified to any width, height

def generate_cropped_images(out_directory: str, crop_instructions: list, min_size: tuple=MIN_SIZE):
    """
    Main function for cropper.py. Generates the cropped images from an list crop_instructions
    out_directory should be the path as a string to where the output files should be placed
    crop_instructions will be a list of tuples in the format of:

    [(filename, index, x-left, y-top, x-right, y-bottom), ...]

    min_size is a tuple in the format (min_width_pixes, min_height_pixels)
    """

    # Iterate through every tuple in crop_instructions
    for instruction in crop_instructions:

        # Save the index number for filename
        index = instruction[1]

        # Save the coordinates
        coords ={
            "left": instruction[2],
            "top": instruction[3],
            "right": instruction[4],
            "bottom": instruction[5]
        }

        # Only process the picture if its bigger than a certain size.
        min_width, min_height = min_size
        if (min_height <= coords["bottom"] - coords["top"] and \
            min_width <= coords["right"] - coords["left"]):

            # Open the file, convert to RGB (handles JPG and PNG)
            main_image = Image.open(instruction[0]).convert("RGB")

            # Save the filename without the path

            filename = instruction[0].split('/')[-1]
            name, extension = filename.rsplit('.')

            # Crop the image using the given coordinates
            cropped_image = main_image.crop((coords["left"], coords["top"], \
                coords["right"], coords["bottom"]))
            path_to_save_file = out_directory + name + "_{}.".format(index) + extension

            # Save the new file
            cropped_image.save(path_to_save_file, "JPEG")
            print("image has been saved")
