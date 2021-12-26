"""
Title::

    pytorch_training.py 

Description::

    this program sets the appropiate envionment hyper-parameters and then loads the
    selected CNN model (implemented on the CNN_models.py file) in order to train it using
    the synthetic data (thermal / night-vision simulated wildfire images) previosly created 
    using the Unreal Engine editor. Once the model gets trained, a summary file is created
    including the trained CNN architecure blueprint and other performance data.

Inputs::

    STDIN data: Model & Dataset versions

Output::

    Tarball file "$WORKSPACE$/bin/CNN/cnn-training_%TIMESTAMP%.tar.gz" including:

    - Trained model binary file -> trained-model.pth
    - Training summary report -> trained-model.info
    - Loss function plot -> loss-curve.png
    - Accuraccy progression plot -> epoch-accuracies.png
    - Label predictions image grid -> labels-prediction.png

Original author::

    Microsoft Docs - Windows AI - Windows Machine Learning

Modified / adapted by:   

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
    2 - https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/ 
    3 - https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1

TODO LIST: 

    - Define training+validation data split -> https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    - track training time and add to report 
    - investigate why sometimes the predicted labels on the summary are way far below the best prediction accuracy. 
      Maybe the microsoft algorythm is buggy and is saving the FIRST trained model (first epoch) instead of the BEST 
      trained model (which uses to provide the worse accuracy results).
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
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# CNN model & Dataset versions
MODEL_VERSION = input("Set the model version you want to train (1 = v1.0, 2 = v2.0, 3 = v3.0): ")
DATASET_VERSION = input("Set the dataset version you want to use to train the model (4 = v4.0 / 5 = v5.0 / 6 = v6.0): ")

# model storage path
MODEL_PATH = "bin/CNN/cnn-training_" + str(TIMESTAMP)

# Dataset paths - DEBUG: replace abs for rel paths?
ROOT_DATA_DIR = 'src/CNN/data/'
TRAIN_DATA_DIR = os.path.abspath(ROOT_DATA_DIR + 'v' + DATASET_VERSION + '.0/training+validation/')  
TEST_DATA_DIR = os.path.abspath(ROOT_DATA_DIR + 'v' + DATASET_VERSION + '.0/test/') 

# opening / creating the file where to store the results
os.mkdir(os.path.abspath(MODEL_PATH))
OUT_FILE = open(os.path.abspath(MODEL_PATH + "/trained-model.info"), "w")

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
BATCH_SIZE = 40

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
####                                      2. IMPORTING THE DATA                                ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

###################################### 2.1 - DATA TRANSFORMATIONS #################################

# Define transformations for the training and test subsets
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

# Define augmentation techniques (mirroring) to increase the number of samples / images on the training and test subsets
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

################################## 2.2 - LOADING TRAINING & TEST DATA #############################

# Create an instance for training. 
train_data = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transformations) + datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=augmentations)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# .info file (output data)
print(" DATASET TOTALS:\n", file=OUT_FILE)
print(" - The number of images in a training set is: ", len(train_loader)*BATCH_SIZE, file=OUT_FILE)

# Create an instance for testing
test_data = datasets.ImageFolder(root=TEST_DATA_DIR, transform=transformations) + datasets.ImageFolder(root=TEST_DATA_DIR, transform=augmentations)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(" - The number of images in a test set is: ", len(test_loader)*BATCH_SIZE, file=OUT_FILE)

print(" - The number of batches per epoch is: ", len(train_loader), file=OUT_FILE)
classes = ('high-intensity-wildfire', 'low-intensity-wildfire', 'no-wildfires')

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                   3. DEFINE THE CNN STRUCTURE                             ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

from CNN_models import *

def set_model_version(input):
    if (input == '1'):
        selection = Network_v1()
        
    elif (input == '2'):
        selection = Network_v2()

    elif (input == '3'): 
        selection = Network_v3()

    return selection

# Instantiate the selected neural network model class imported from the src/CNN/CNN_Models.py file
model = set_model_version(MODEL_VERSION)

print("\n***********************************************\n\n",
      "CNN BLUEPRINT:\n\n",  
      model,
      "\n\n***********************************************\n",
      file=OUT_FILE)

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                    4. DEFINE THE LOSS FUNCTION                            ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

# Define a loss function
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001) # setting lr = 0.0001 DO NOT increases accuracy, and training time is more or less the same 

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                        5. TRAIN THE MODEL                                 ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

import torch
from torch.autograd import Variable

# Function to save the model
def saveModel():
    torch.save(model.state_dict(), MODEL_PATH + "/trained-model.pth")

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
    plt.savefig(MODEL_PATH + '/loss-curve.png')
    plt.show()

    # ...as well as the accuracy progression
    plt.plot(accuracies, color='purple')
    plt.xlabel("training epoch")
    plt.ylabel("model accuracy")
    plt.title("MODEL ACCURACY PROGRESSION")
    plt.legend("MODEL ACCURACY PROGRESSION")
    plt.savefig(MODEL_PATH + '/epoch-accuracies.png')
    plt.show()

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                6. TEST THE MODEL ON THE TEST DATA                         ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(MODEL_PATH + '/labels-prediction.png')
    plt.show()

# Function to test the model with a batch of images and show the labels predictions

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
    # DEBUG - The original code does not have a testModelAccuracy() method. 
    #testModelAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "/trained-model.pth"))

    # Test with batch of images
    testBatch()
    OUT_FILE.close()

    # Generating final results report tarball file

    import tarfile
    #import os.path
    import shutil

    archive = tarfile.open(MODEL_PATH+".tar.gz", "w|gz")
    archive.add(MODEL_PATH, arcname="")
    archive.close()

    print("\nTraining results file -> " + os.path.abspath(MODEL_PATH) + ".tar.gz\n")
    shutil.rmtree(MODEL_PATH)




