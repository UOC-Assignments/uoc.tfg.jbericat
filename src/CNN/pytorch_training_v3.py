"""
title:: 


description::


inputs::


output::


original author::
    Microsoft Docs - Windows AI - Windows Machine Learning

Modified / adapted by:    
    Jordi Bericat Ruz - Universitat Oberta de Catalunya

references::
    1 - https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
"""

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                       1. PARAMETRIZATION                                  ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

####################################### 1.1 - GLOBAL PARAMETERS ###################################

import os
import time

# This script's execution timestamp
EXEC_TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# CNN model & Dataset versions
MODEL_VERSION = 2
DATASET_VERSION = input("Type the dataset version you want to train the model against (4 = v4.0 / 5 = v5.0 / 6 = v6.0): ")

# model storage path
MODEL_BIN_PATH = "bin/CNN/CNN-Model-v" + str(MODEL_VERSION) + "_dataset-v" + str(DATASET_VERSION) + "_" + str(EXEC_TIMESTAMP)

# Dataset paths - DEBUG: replace abs for rel paths?
ROOT_DATA_DIR = 'src/CNN/data/'
TRAIN_DATA_DIR = os.path.abspath(ROOT_DATA_DIR + 'v' + DATASET_VERSION + '.0/training+validation/')  
TEST_DATA_DIR = os.path.abspath(ROOT_DATA_DIR + 'v' + DATASET_VERSION + '.0/test/') 

# opening / creating the file where to store the results
OUT_FILE = open(MODEL_BIN_PATH + ".info", "w")

###################################### 1.2 - DATA-BOND PARAMETERS ##################################

DATASET_IMG_SIZE = 229

# The number of labels correspond to the number of classes we defined on previous
# stages of this project. To sum-up, we have: 
# 
# - Class #1: High-intensity wildfires 
# - Class #2: Medium-intensity wildfires 
# - Class #3: Low-intensity wildfires
# - Class #4: Images with no wildfires at all
NUMBER_OF_LABELS = OUTPUT_FEATURES = 3 
IMG_CHANNELS = 1

###################################### 1.4 - TRAINING PARAMETERS ####################################

# We can set as many epochs as desired, considering that there is a threshold when the 
# model stops improving it's performance after each training iteration (plotting the 
# loss function at the end of the training process could be useful to optimize the training 
# time vs perfomance balance.
EPOCHS = 5

# TFGthe PoC's dataset consists of 500x2=1000 training images and 200x2=400 test images (we're adding the augmented dataset). 
# Hence, we define a batch size of X to load YY & ZZ batches of images respectively on each epoch:
BATCH_SIZE = 10

####################################### 1.5 - OUTPUT TO SUMMARY FILE #################################

# storing all the training environment on the results .info file's preface
OUT_FILE.writelines([ "***********************************************\n",
                      "***     PoC's CNN Model Training Summary    ***\n", 
                      "***********************************************\n", 
                      "              Model version: v" + str(MODEL_VERSION) + ".0\n",
                      "             Dataset version: v" + str(DATASET_VERSION) + ".0\n",
                      "***********************************************\n\n",
                      " DATA PARAMETERS:\n\n",
                      " - Image size = (" + str(DATASET_IMG_SIZE) + " x " + str(DATASET_IMG_SIZE) + " x " + str(IMG_CHANNELS) + ")\n",
                      " - Number of classes / labels = "  + str(NUMBER_OF_LABELS) + "\n",
                      " - Batch size = "  + str(BATCH_SIZE) + "\n\n",
                      "***********************************************\n\n"                     
                      ])


###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                       1. IMPORTING THE DATA                               ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

# https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1.1 - Loading and normalizing the data.
# Define transformations for the training and test sets

transformations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL (GRAYSCALE)
    # We need square images to feed the model (the raw dataset has 640x512 size images)
    # DEBUG - UNCOMMENT NEXT LINE FOR v4 DATASET
    #transforms.RandomResizedCrop(512),
    # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

augmentations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL (GRAYSCALE)
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation techique: Horizontal Mirroring
    # We need square images to feed the model (the raw dataset has 640x512 size images)
    # DEBUG - UNCOMMENT NEXT LINE FOR v4 DATASET
    #transforms.RandomResizedCrop(512),
    # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
    transforms.Resize(DATASET_IMG_SIZE)
])

# 1.2 - Create an instance for training. 
train_data = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transformations) + datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=augmentations)

# 1.3 - Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# .info file data
print(" DATASET TOTALS:\n", file=OUT_FILE)
print(" - The number of images in a training set is: ", len(train_loader)*BATCH_SIZE, file=OUT_FILE)

# 1.4 - Create an instance for testing
test_data = datasets.ImageFolder(root=TEST_DATA_DIR, transform=transformations) + datasets.ImageFolder(root=TEST_DATA_DIR, transform=augmentations)

# 1.5 - Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(" - The number of images in a test set is: ", len(test_loader)*BATCH_SIZE, file=OUT_FILE)

print(" - The number of batches per epoch is: ", len(train_loader), file=OUT_FILE)
classes = ('high-intensity-wildfire', 'medium-intensity-wildfire', 'low-intensity-wildfire ', 'no-wildfire')


###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                     2. DEFINE THE CNN STRUCTURE                           ####
####                                                                                           ####   
###################################################################################################
###################################################################################################


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from CNN_models import *

# Instantiate the selected neural network model class imported from file CNN_Models.py)
model = Network_v1()
print("\n***********************************************\n\n",
      "CCN STRUCTURE:\n\n",  
      model,
      "\n\n***********************************************\n",
      file=OUT_FILE)
# 2.2 - Define a loss function
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001) # setting lr = 0.0001 increases accuracy, but reduces training time significantly


###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                         3. TRAIN THE MODEL                                ####
####                                                                                           ####   
###################################################################################################
###################################################################################################


from torch.autograd import Variable

# Function to save the model
def saveModel():
    torch.save(model.state_dict(), MODEL_BIN_PATH + ".pth")

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader: # BUGFIX PR #126 ->  See TASK#05.6: https://github.com/UOC-Assignments/uoc.tfg.jbericat/issues/96
            images, labels = images.cuda(), labels.cuda() # BUGFIX PR #126 ->  See TASK#05.6: https://github.com/UOC-Assignments/uoc.tfg.jbericat/issues/96
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # INFO
    print("\nTraining the model and creating the summary, hold-on tight...\n")

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" TRAINING STATS:\n\n", 
        "Model trained on", device, "device\n",file=OUT_FILE)
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # In order to plot the losses we need a data structure to accumulate each epoch's loss value
    # https://discuss.pytorch.org/t/plotting-loss-curve/42632/2
    losses = []
    accuracies = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000), file=OUT_FILE)
                # zero the loss
                running_loss = 0.0
        
        # Here we need to save this epoch's running loss, so we can plot it later
        losses.append(running_loss / len(train_data))

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print(' For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy),file=OUT_FILE)
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

        # accumulating accuracies to draw the plot
        accuracies.append(accuracy)

    # Now we can plot the loss curve... 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html 
    plt.plot(losses, color='red')
    plt.xlabel("training epoch")
    plt.ylabel("running loss")
    plt.title("LOSS CURVE")
    plt.legend("LOSS CURVE")
    plt.savefig(MODEL_BIN_PATH + '_loss-curve.png')
    plt.show()

    # ...as well as the accuracy progression
    plt.plot(accuracies, color='purple')
    plt.xlabel("training epoch")
    plt.ylabel("model accuracy")
    plt.title("MODEL ACCURACY PROGRESSION")
    plt.legend("MODEL ACCURACY PROGRESSION")
    plt.savefig(MODEL_BIN_PATH + '_epoch-accuracies.png')
    plt.show()


###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                4. TEST THE MODEL ON THE TEST DATA                         ####
####                                                                                           ####   
###################################################################################################
###################################################################################################


import matplotlib.pyplot as plt
import numpy as np

# 4.1 - Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 4.2 - Function to test the model with a batch of images and show the labels predictions

def testBatch():

    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('\nReal labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(BATCH_SIZE)),file=OUT_FILE)
  
    # Let's see what if the model identifies the  labels of these example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('\nPredicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(BATCH_SIZE)),file=OUT_FILE)


###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                            6. MAIN CODE                                   ####
####                                                                                           ####   
###################################################################################################
###################################################################################################


if __name__ == "__main__":
    
    # Let's build our model
    train(EPOCHS)
    print('Finished Training')

    # Test which classes performed well
    # DEBUG - UNCOMMENT NEXT LINE! 
    #testModelAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network_v2()
    model.load_state_dict(torch.load(MODEL_BIN_PATH + ".pth"))

    # Test with batch of images
    testBatch()

    # RESULTS REPORTS
    print("\nTrained model binary file -> " + os.path.abspath(MODEL_BIN_PATH) + ".pth\n")
    print("Training summary file -> " + os.path.abspath(MODEL_BIN_PATH) + ".info\n")
    print("Loss function curve -> " + os.path.abspath(MODEL_BIN_PATH) + "_loss-curve.png\n")
    print("Epoch accuracy stats -> " + os.path.abspath(MODEL_BIN_PATH) + "_epoch-accuracy.png\n")