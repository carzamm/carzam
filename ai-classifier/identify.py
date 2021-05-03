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

# This takes all the directory names and sets them to equal the class indexes for the model
def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# This is a custom function that will go through every file in ./input/verify and just see if the model
# comes up with the right answer.
def test_all_cars():
    for file in os.listdir(verification_dir):
        filename = os.fsdecode(file)

        # transforms for the input image
        loader = transforms.Compose([transforms.Resize((400, 400)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        image = Image.open(verification_dir+filename)
        image = loader(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        image = image.cuda()
        output = model_ft(image)
        conf, predicted = torch.max(output.data, 1)

        classes, c_to_idx = find_classes(dataset_dir+"train")

        # get the class name of the prediction
        print("\nSupposed to be {}".format(filename))
        print(classes[predicted.item()], "confidence: ", conf.item())

if __name__ == "__main__":

    # This part detecs if the device has a GPU capable of running CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Clear the CUDA Memory
    # print(torch.cuda.memory_summary(device=0, abbreviated=False))

    # Define where the data is held
    dataset_dir = "./input/"

    # This specifies the model 'Resnet34' which is pretrained, we only want to change the last layer
    # which is the output layer.
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # Freeze all the layers of the model since it is pretrained, we only want to update the last layer
    for param in model_ft.parameters():
        param.requires_grad = False

    # replace the last fc layer with an untrained one (requires grad by default)
    model_ft.fc = nn.Linear(num_ftrs, QTY_CLASSES)
    model_ft = model_ft.to(device)

    # Load the saved weights so we can utilize our pre-trained data for vehicles
    MODEL_FILENAME = "saved_model.pt"
    loaded_weights = torch.load(MODEL_FILENAME)
    model_ft.load_state_dict(loaded_weights)

    # switch the model to evaluation mode to make dropout and batch norm work in eval mode
    model_ft.eval()

    # evaluate the cars in the verification_dir
    verification_dir=dataset_dir+"verify/"
    test_all_cars()