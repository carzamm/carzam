"""
Project: Carzam - CS 467 Capstone
Filename: train.py
Description: AI Vehicle Classifier - Train and Save Model
Source: Origin of code template is from 
    https://www.kaggle.com/deepbear/pytorch-car-classifier-90-accuracy

To Run:     python3 train.py [clean dataset dir in /data]
Example:    python3 train.py ready
"""

# Python Imports
import time
import os
import sys
import argparse
import csv

# MatPlotLib Imports
import matplotlib.pyplot as plt

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# This takes all the directory names and sets them to equal the class indexes for the model
def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# Plots the results of training - used when the AI is done a training evolution
def plot_ai_results(training_losses, training_accs, test_accs):
    f, axarr = plt.subplots(2,2, figsize = (12, 8))
    axarr[0, 0].plot(training_losses)
    axarr[0, 0].set_title("Training loss")
    axarr[0, 1].plot(training_accs)
    axarr[0, 1].set_title("Training acc")
    axarr[1, 0].plot(test_accs)
    axarr[1, 0].set_title("Test acc")
    plt.show()

# This is the function that trains the model
def train_model(model, criterion, optimizer, scheduler, n_epochs):
    
    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        print("Training Epoch - {} of {}".format(epoch, n_epochs))
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            #inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_duration = time.time()-since
        epoch_loss = running_loss/len(trainloader)
        epoch_acc = 100/32*running_correct/len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
        
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_acc = eval_model(model)
        test_accuracies.append(test_acc)
        
        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
        since = time.time()
    print('Finished Training')
    return model, losses, accuracies, test_accuracies

# This is the function that evaluates the model, this occurs after every training iteration
def eval_model(model):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            #images = images.to(device).half() # uncomment for half precision model
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc

# Writes the CSV file
def write_csv(train_loss, train_acc, test_acc):
    with open("results.csv", "a") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Batch Size",
                        "Learning Rate",
                        "Momentum",
                        "Test Accuracy",
                        "Training Loss",
                        "Training Accuracy"])
        for i in range(0, len(train_acc)):
            writer.writerow([BATCH_SIZE,
                            LEARNING_RATE,
                            MOMENTUM,
                            test_acc[i],
                            train_loss[i],
                            train_acc[i]])

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Trains the Vehicle Identification AI.')

    parser.add_argument('clean', help='Name of Directory of Clean Dataset in /data')
    parser.add_argument('-b', default=32, type=int, help='Batch Size')
    parser.add_argument('-e', default=10, type=int, help='Number of Training Epochs')
    parser.add_argument('-c', default=9, type=int, help='Number of Different Car Classes in Dataset')
    parser.add_argument('-l', default=0.01, type=float, help='Learning Rate')
    parser.add_argument('-m', default=0.9, type=float, help='Momentum')
    parser.add_argument('-w', action="store_true", help='Write results to "results.csv"')
    parser.add_argument('-p', action='store_true', help='Plot resutls of training at completion.')
    args = parser.parse_args()

    # Define some constants to finetune hyperparameters
    BATCH_SIZE = args.b
    TRAINING_EPOCHS = args.e
    QTY_CLASSES = args.c # THIS MUST BE THE NUMBER OF POSSIBLE OUTPUTS (MAKE/MODEL combinations)

    LEARNING_RATE = args.l # Getting +80% accuracy w/ 0.01 <= LR <= 0.0175
    MOMENTUM = args.m # Seems like the default for this should be 0.9 >= momentum 0.8

    # This part detecs if the device has a GPU capable of running CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Clear the CUDA Memory
    # print(torch.cuda.memory_summary(device=0, abbreviated=False))

    # Define where the data is held
    dataset_dir = os.path.join("../data", sys.argv[1])

    # This specifies how the data will be transformed from the image to what PyTorch will recognize
    # these values are specific and shouldn't be changed. They were specified by the PyTorch documentation.

    train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # This defines the dataset location and then loads the data
    dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"/train", transform = train_tfms)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 2)

    dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir+"/test", transform = test_tfms)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size = BATCH_SIZE, shuffle=False, num_workers = 2)


    # This specifies the model 'Resnet34' which is pretrained, we only want to change the last layer
    # which is the output layer.
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # Freeze all the layers of the model since it is pretrained, we only want to update the last layer
    # for param in model_ft.parameters():
    #     param.requires_grad = False

    # replace the last fc layer with an untrained one (requires grad by default)
    model_ft.fc = nn.Linear(num_ftrs, QTY_CLASSES)
    model_ft = model_ft.to(device)

    # This specifies how the model is optimized between epochs such as learning rate and momentum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # This line specifically handles learning rate of the AI and causes it to adjust when there is a 'learning plateau'.
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

    # This is the call to train the actual model: It returns the model with some data dervied from training
    model_ft, training_losses, training_accs, test_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=TRAINING_EPOCHS)

    # This plots the results
    if args.p:
        plot_ai_results(training_losses, training_accs, test_accs)

    # Save the CSV file if requested
    if args.w:
        write_csv(training_losses, training_accs, test_accs)

    # Save the model to the current directory
    MODEL_FILENAME = "saved_model.pt"
    torch.save(model_ft.state_dict(), MODEL_FILENAME)