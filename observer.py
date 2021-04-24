"""
Project: Carzam - CS 467 Capstone
Filename: observer.py
Description: Watches a directory for new files. When a new file appears it is processed
by 'recognizer' and 'cropper'
"""

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cropper import generate_cropped_images
from recognizer import recognize_objects


# Create a custom event handler for new files in a directory
class NewFileEvent(FileSystemEventHandler):
    """
    This is an override of the FileSystemEventHandler from watchdog. We are speicfically
    overriding the on_created event since all we care about is when a file is created.
    """
    def __init__(self, out_directory: str):
        super(FileSystemEventHandler)
        self.out_directory = out_directory

    # Override the on_created event to handle new images when they appear
    def on_created(self, event):

        # Get the path of the file
        filename = event.src_path

        # Check the file size
        filesize = os.path.getsize(filename)

        # We will loop, checking the file size in the case its being uploaded it will
        # start at 0 and grow, we wait for it to stabilize at a final value
        same = 0

        print("Waiting for file to be fully written...")
        while same < 4:

            # Sleep for 0.3 seconds between checking the file size, not the best but works
            time.sleep(0.3)

            # If the file size isnt zero and the old size matches the new one
            if filesize != 0 and filesize == os.path.getsize(filename):

                # Increment same counter
                same += 1

            # Otherwise the file size changed and we set same back to 0
            else:
                filesize = os.path.getsize(filename)
                same = 0
        print("File appears to be fully written...")


        # Search the file for objects
        crop_instructions = recognize_objects(filename)
        generate_cropped_images(self.out_directory, crop_instructions)


# Acts as a constructor for an observer object created with the 'watchdog' library
def directory_observer(path_to_watch: str, path_to_write: str):
    """
    directory_observer takes two arguments as strings, the path_to_watch which
    is a directory for new files to be added to and path_to_write, which is
    where the output files that are cropped will be written to
    """

    # Create our custom new file event handler
    new_file_handler = NewFileEvent(path_to_write)

    # Create an Observer object from watchdog
    observer = Observer()

    # Add the file handler and path to watch to the Observer object
    observer.schedule(new_file_handler, path_to_watch)

    # Return the object to main to be run as needed
    return observer
