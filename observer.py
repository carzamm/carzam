"""
Project: Carzam - CS 467 Capstone
Filename: observer.py
Description: Watches a directory for new files. When a new file appears it is processed by 'recognizer' and 'cropper'
"""

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cropper import generate_cropped_images
from recognizer import recognize_objects


# Create a custom event handler for new files in a directory
class NewFileEvent(FileSystemEventHandler):
    def __init__(self, out_directory: str):
        super(FileSystemEventHandler)
        self.out_directory = out_directory

    # Override the on_created event to handle new images when they appear
    def on_created(self, event):
        
        # Get the path of the file
        filename = event.src_path

        # Search the file for objects
        crop_instructions = recognize_objects(filename)
        generate_cropped_images(self.out_directory, crop_instructions)


# Acts as a constructor for an observer object created with the 'watchdog' library
def directory_observer(path_to_watch: str, path_to_write: str):

    # Create our custom new file event handler
    new_file_handler = NewFileEvent(path_to_write)

    # Create an Observer object from watchdog
    observer = Observer()

    # Add the file handler and path to watch to the Observer object
    observer.schedule(new_file_handler, path_to_watch)

    # Return the object to main to be run as needed
    return observer
