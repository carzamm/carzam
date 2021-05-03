import os

DATA_DIR = "./ai-classifier/output"


def count():
    status = {}
    top_level_folders = os.listdir(DATA_DIR)
    for folder in top_level_folders:
        status[folder] = {}
        if folder in ['train','test']:
            for car_class in os.listdir(os.path.join(DATA_DIR, folder)):
                if not os.path.isfile(os.path.join(DATA_DIR, folder, car_class)):
                    status[folder][car_class] = len(os.listdir(os.path.join(DATA_DIR, folder, car_class)))
        if folder in ['verify']:
            status[folder] = len(os.listdir(os.path.join(DATA_DIR, folder))) 
    return status

def calculate_class_size(data: dict):
    
    classes = list(data.keys())
    return len(status[classes[0]])

def display_row(row: tuple):
    
    print("{:40}|{:12}|{:12}|{:12}|{:12.1f}%".format(*row))

def display_results(data: dict):
    
    cars = data['train'].keys()
    for car in cars:
        name = car
        total = data['train'][car] + data['test'][car]
        test = data['test'][car]
        train = data['train'][car]
        ratio = float(train/total * 100)
        display_row((name, total, train, test, ratio))

def calculate_total_images(data: dict):
    cars = data['train'].keys()
    total = 0
    for car in cars:
        total += data['train'][car] + data['test'][car]
    return total + data['verify']


def display_header(data: dict):

    header = "Class", "Total Files", "Train", "Test", "Ratio"
    print("{:40}|{:12}|{:12}|{:12}|{:12}%".format(*header))
    print("-" * 92)

def display_footer(data: dict):
    
    print("-" * 92)
    print("Number of Classes: {}".format(calculate_class_size(data)))
    print("Verification Images: {}".format(data["verify"]))
    print("Total Images: {}".format(calculate_total_images(data)))




if __name__ == "__main__":

    status = count()
    display_header(status)
    display_results(status)
    display_footer(status)