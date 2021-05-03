"""
Project: Carzam - CS 467 Capstone
Filename: identify.py
Description: AI Vehicle Classifier - Train and Save Model
Source: Origin of code template is from https://www.kaggle.com/deepbear/pytorch-car-classifier-90-accuracy
"""
import time
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import PIL.Image as Image

QTY_CLASSES = 9 # THIS MUST BE THE NUMBER OF POSSIBLE OUTPUTS (MAKE/MODEL combinations)
MODEL_FILENAME = "saved_model.pt" # LOCATION OF THE SAVED WEIGHTS & BIASES

class Identifier:
    def __init__(self):
        
        # This part detecs if the device has a GPU capable of running CUDA
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        # Clear the CUDA Memory
        # print(torch.cuda.memory_summary(device=0, abbreviated=False))

        # Define where the data is held
        self.dataset_dir = "./input/"
        self.verification_dir=self.dataset_dir+"verify/"

        # This specifies the model 'Resnet34' which is pretrained, we only want to change the last layer
        # which is the output layer.
        self.model_ft = models.resnet34(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features

        # Freeze all the layers of the model since it is pretrained, we only want to update the last layer
        for param in self.model_ft.parameters():
            param.requires_grad = False

        # replace the last fc layer with an untrained one (requires grad by default)
        self.model_ft.fc = nn.Linear(self.num_ftrs, QTY_CLASSES)
        self.model_ft = self.model_ft.to(self.device)

        # Load the saved weights so we can utilize our pre-trained data for vehicles

        self.loaded_weights = torch.load(MODEL_FILENAME)
        self.model_ft.load_state_dict(self.loaded_weights)

        # switch the model to evaluation mode to make dropout and batch norm work in eval mode
        self.model_ft.eval()


    # This takes all the directory names and sets them to equal the class indexes for the model
    def find_classes(self, dir):
        classes = os.listdir(dir)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    # This is a custom function that will go through every file in ./input/verify and just see if the model
    # comes up with the right answer.
    def test_all_cars(self, list_of_paths=None):

        # Using default variable list_of_paths. Set to None, but can be given a list of paths to 'test all cars'
        if list_of_paths is None:
            files = [self.verification_dir + x for x in os.listdir(self.verification_dir)]
        else:
            files = list_of_paths
        
        # Used to store results of vehicle recognition
        results = []

        # Identify each file
        for file in files:
            results.append(self.test_single_car(file))

        return results



    
    def test_single_car(self, path: str):
            filename = os.fsdecode(path)

            # transforms for the input image
            loader = transforms.Compose([transforms.Resize((400, 400)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            image = Image.open(path)
            image = loader(image).float()
            image = torch.autograd.Variable(image, requires_grad=True)
            image = image.unsqueeze(0)
            image = image.cuda()
            output = self.model_ft(image)
            conf, predicted = torch.max(output.data, 1)

            # classes, c_to_idx = self.find_classes(self.dataset_dir+"train")
            classes = [
                'Acura TL', 'Audi A4', 'BMW X3', 'Buick Enclave', 'Cheverolet Silverado 1500',
                'Ford Crown Victoria', 'Ford Fusion', 'Jeep Wrangler Unlimited', 'Toyota RAV4'
            ]

            c_to_idx = {
                'Acura TL': 0,
                'Audi A4': 1,
                'BMW X3': 2,
                'Buick Enclave': 3,
                'Chevrolet Silverado 1500': 4,
                'Ford Crown Victoria': 5,
                'Ford Fusion': 6,
                'Jeep Wrangler Unlimited': 7,
                'Toyota RAV4': 8
            }

            # get the class name of the prediction
            print("\nSupposed to be {}".format(filename))
            print(classes[predicted.item()], "confidence: ", conf.item())

            return (path, classes[predicted.item()], conf.item())


if __name__ == "__main__":
    i = Identifier()
    i.test_all_cars()