from observer import directory_observer

OUT_DIRECTORY = "./data/outimages/"
IN_DIRECTORY = "./data/testimages/"

observer = directory_observer(IN_DIRECTORY, OUT_DIRECTORY)
observer.start()
observer.join()