"""
Project: Carzam - CS 467 Capstone
Filename: carzam.py
Description: The main file for carzam. Creates an observer that watches an input directory, 
and then waits for new files to be added, processing them as they are
"""


from observer import directory_observer

OUT_DIRECTORY = "./data/outimages/"
IN_DIRECTORY = "./data/testimages/"

observer = directory_observer(IN_DIRECTORY, OUT_DIRECTORY)
observer.start()
observer.join()