from recognizer import recognize_objects
from cropper import generate_cropped_images

path_to_file = "./data/testimages/2020-acura-mdx.jpg"
out_directory = "./data/outimages/"

crop_instructions = recognize_objects(path_to_file)
generate_cropped_images(out_directory, crop_instructions)