"""
Project: Carzam - CS 467 Capstone
Filename: carzam.py
Description: The main file for carzam. Creates an observer that watches an input directory,
and then waits for new files to be added, processing them as they are
"""

from pathlib import Path
from observer import directory_observer

# CONSTANTS - CAN BE CHANGED
OUT_DIRECTORY = "./data/outimages/"
IN_DIRECTORY = "./data/testimages/"

# Make the directories declared as constants if they do not exist
Path(IN_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(OUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Create the observer to scan IN_DIRECTORY and then output to OUT_DIRECTORY
observer = directory_observer(IN_DIRECTORY, OUT_DIRECTORY)
observer.start()
observer.join()
