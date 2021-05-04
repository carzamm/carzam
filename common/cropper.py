"""
Project: Carzam - CS 467 Capstone
Filename: cropper.py
Description: Single function that accepts a list of crop_instructions in the format

[(filename, index, x-left, y-top, x-right, y-bottom), ...]

and outputs cropped image files using those coordinates

MIN_SIZE constant is defined to filter out very small images. It's a tuple of 
width in pixels by height in pixels. Images that are smaller than this on EITHER
side will not be processed.

Catches warnings about improperly formatted files and displays them to the console.
"""


from os import path
from hashlib import sha256
import warnings
from PIL import Image

MIN_SIZE = 400, 400 # This can be modified to any width, height

# Override the warnings from showing, they're not very helpful
def ignore():
    return

def generate_cropped_images(out_directory: str, crop_instructions: list, min_size: tuple=MIN_SIZE, padding: bool=False):
    """
    Main function for cropper.py. Generates the cropped images from an list crop_instructions
    out_directory should be the path as a string to where the output files should be placed
    crop_instructions will be a list of tuples in the format of:

    [(filename, index, x-left, y-top, x-right, y-bottom), ...]

    min_size is a tuple in the format (min_width_pixes, min_height_pixels)
    """
    crypt = sha256()
    output = False  # We use this to tell if the generator did anyting

    # If there are image errors with the PIL, we can now catch them.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.formatwarning = ignore

        # List object that holds strings which point to the cropped files
        cropped_files = []

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

                # Generator will write a file
                output = True

                # Open the file, convert to RGB (handles JPG and PNG)
                try:
                    main_image = Image.open(instruction[0]).convert("RGB")
                except UserWarning:
                    print("Issue with {}. Skipping...".format(instruction[0]))
                    return (False, [])

                if main_image:

                    # Get the original image size
                    o_width, o_height = main_image.size

                    # Add padding if we want it
                    if padding:
                        padding = 20 # pixels
                        coords["bottom"] = min(coords["bottom"] + padding, o_height)
                        coords["top"] = max(coords["top"] - padding, 0)
                        coords["right"] = min(coords["right"] + padding, o_width)
                        coords["left"] = max(coords["left"] - padding, 0)

                    # Save the filename without the path
                    filename = instruction[0].split('/')[-1]

                    # Tries to use the existing filename, if it has weird characters
                    # we just give it a random alphanumeric string as filename
                    try:
                        name, extension = filename.rsplit('.')
                    except ValueError:
                        crypt.update(bytes(filename, 'utf-8'))
                        name = crypt.hexdigest()
                        extension = "jpg"

                    # Crop the image using the given coordinates
                    cropped_image = main_image.crop((coords["left"], coords["top"], \
                        coords["right"], coords["bottom"]))

                    out_file = name + "_{}.".format(index) + extension
                    out_path = path.join(out_directory, out_file)

                    # Save the new file
                    cropped_image.save(out_path, "JPEG")
                    cropped_files.append(out_path)

    return output, cropped_files

