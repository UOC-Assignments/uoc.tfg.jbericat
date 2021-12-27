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

Python env: TODO Python 6.4.6 + pytorch x.y....... 

Original author::

    Microsoft Docs - Windows AI - Windows Machine Learning

Modified / adapted by:   

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
    2 - https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/ 
    3 - https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1
    4 - https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    5 - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html  
    6 - https://discuss.pytorch.org/t/plotting-loss-curve/42632/2

TODO LIST: 

    - Define training+validation data split -> https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
        -> "Using PyTorch’s random_split function, we can easily split our data."
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

# CNN model & Dataset versions (we do not waste time implementing error control)
MODEL_VERSION = input("Set the model version you want to train (1 = v1.0, 2 = v2.0, 3 = v3.0): ")
DATASET_VERSION = input("Set the dataset version you want to use to train the model (4 = v4.0 / 5 = v5.0 / 6 = v6.0 / 7 = v7.0): ")

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
# - Class #2: Medium-intensity wildfires -- REMOVED FROM DATASET v7.0!!!
# - Class #3: Low-intensity wildfires
# - Class #4: Images with no wildfires at all
NUMBER_OF_LABELS = OUTPUT_FEATURES = 3
IMG_CHANNELS = 1

###################################### 1.4 - TRAINING PARAMETERS ####################################

# We can set as many epochs as desired, considering that there is a threshold when the 
# model stops improving it's performance after each training iteration (plotting the 
# loss function at the end of the training process could be useful to optimize the training 
# time vs perfomance balance.
EPOCHS = 10

# TFGthe PoC's dataset consists of 500x2=1000 training images and 200x2=400 test images (we're adding the augmented dataset). 
# Hence, we define a batch size of X to load YY & ZZ batches of images respectively on each epoch:
BATCH_SIZE = 128

# Learning rate: 
LEARNING_RATE = 0.00001

############################### 1.5 - OUTPUT SUMMARY DATA (cnn-training.info) #####################

# storing all the training environment on the results .info file's summary
OUT_FILE.writelines([ "********************************************************************************\n",
                      "***                                                                          ***\n",
                      "***                      PoC's CNN Model Training Summary                    ***\n",
                      "***                                                                          ***\n",
                      "********************************************************************************\n",
                      "***                                                                          ***\n",
                      "***                             Model version: v" + str(MODEL_VERSION) + ".0" + 26*" " + "***\n",
                      "***                            Dataset version: v" + str(DATASET_VERSION) + ".0" + 25*" " + "***\n"
                      "***                                                                          ***\n",
                      "********************************************************************************\n\n",
                      " TRAINING PARAMETERS:\n\n",
                      " - Image size = (" + str(DATASET_IMG_SIZE) + " x " + str(DATASET_IMG_SIZE) + " x " + str(IMG_CHANNELS) + ")\n",
                      " - Number of classes / labels = "  + str(NUMBER_OF_LABELS) + "\n",
                      " - Batch size = "  + str(BATCH_SIZE) + "\n",
                      " - Learning rate = " + str(LEARNING_RATE) + "\n\n",
                      "********************************************************************************\n\n"                     
                      ])

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                      2. IMPORTING THE DATA                                ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

###################################### 2.1 - DATA TRANSFORMATIONS #################################

# Define transformations for the training and test subsets
transformations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL ONLY (GRAYSCALE)
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
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation technique: Horizontal Mirroring
    # We need square images to feed the model (the raw dataset has 640x512 size images)
    # DEBUG - UNCOMMENT NEXT LINE FOR v4 DATASET
    #transforms.RandomResizedCrop(512),
    # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
    transforms.Resize(DATASET_IMG_SIZE)
])

########################### 2.2 - LOADING TRAINING, VALIDATION & TEST DATA #######################

# Create an instance for training. 
train_data = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transformations) + datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=augmentations)

# SPLITTING TRAINING & VALIDATION DATA
# https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# calculate the train/validation split
print("\n[INFO] generating the train/validation split...\n")
numTrainSamples = int(len(train_data) * TRAIN_SPLIT)
numValSamples = int(len(train_data) * VAL_SPLIT)
(train_data, val_data) = random_split(train_data,
	[numTrainSamples+1, numValSamples], # BUGFIX -> https://fixexception.com/torch/sum-of-input-lengths-does-not-equal-the-length-of-the-input-dataset/
	generator=torch.Generator().manual_seed(42))

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Create a loader for the validation set
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Create an instance for testing
test_data = datasets.ImageFolder(root=TEST_DATA_DIR, transform=transformations) + datasets.ImageFolder(root=TEST_DATA_DIR, transform=augmentations)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# calculate steps per epoch for training and validation set (needed to track training + validation loss & accuracy)
trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(val_loader.dataset) // BATCH_SIZE

############################# 2.3 - OUTPUT SUMMARY DATA (cnn-training.info) #######################

print(" DATASET TOTALS:\n", file=OUT_FILE)
print(" - The number of images in a training set is: ", len(train_loader)*BATCH_SIZE, file=OUT_FILE)
print(" - The number of images in a validation set is: ", len(val_loader)*BATCH_SIZE, file=OUT_FILE)
print(" - The number of images in a test set is: ", len(test_loader)*BATCH_SIZE, file=OUT_FILE)

print(" - The number of batches per epoch is: ", len(train_loader), file=OUT_FILE)
classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

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

# .info file output summary data 
print("\n********************************************************************************\n\n",
      "CNN BLUEPRINT:\n\n",  
      model,
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
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)  

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                        5. TRAIN THE MODEL                                 ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

################################# 5.1 - AUXILIAR FUNCTIONS & GLOBALS ##############################

from torch.autograd import Variable

# Function to store the model on the local drive
def saveModel():
    torch.save(model.state_dict(), MODEL_PATH + "/trained-model.pth")

'''
# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    # for TESTING we need to set the model in evaluation mode
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    # turn off autograd for testing evaluation
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
'''

# Function to test the model with the test dataset and print the accuracy for the test images
def calculateAccuracy(data_loader):
    
    # for TESTING we need to set the model in evaluation mode
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    # turn off autograd for testing evaluation
    with torch.no_grad():
        for images, labels in data_loader: # BUGFIX PR #126 ->  See TASK#05.6: https://github.com/UOC-Assignments/uoc.tfg.jbericat/issues/96
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


################################## 5.2 - MODEL TRAINING FUNCTION #############################

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):


    # initialize a dictionary to store training history
    STATS = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": []
    }
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n********************************************************************************\n\n",
          " MODEL TRAINING STATS:\n\n", "Model trained on", device, "device\n",
          file=OUT_FILE)

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # STDOUT INFO
    print("\n[INFO] Training the model and creating the summary, hold-on tight...\n")

    # measure how long training + validation process is going to take -> https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Waits for everything to start running
    torch.cuda.synchronize()

    # Start recording time (miliseconds)
    start.record()

    # loop over the dataset multiple times
    for epoch in range(num_epochs): 

        #################### 5.2.1 - MODEL TRAINING ####################
        
        # Set the model in training mode
        model.train()
     
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        #trainCorrect = 0
        #valCorrect = 0

        # loop over the training set
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

            # extract the loss and total correct predictions value         
            totalTrainLoss += loss.item()     
            #trainCorrect += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        
        #################### 5.2.2 - MODEL VALIDATION ####################

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # loop over the validation set
            for i, (images, labels) in enumerate(val_loader, 0):

                # get the inputs
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                # make the predictions and calculate the validation loss
                predictions = model(images)
                totalValLoss += loss_fn(predictions, labels)

                # calculate the number of correct predictions
                #valCorrect += (predictions.argmax(1) == labels).type(torch.float).sum().item()

	#################### 5.2.3 - TRAIN & VAL STATS ####################
    
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
        #trainCorrect = trainCorrect / len(train_loader)
        #valCorrect = valCorrect / len(val_loader)
        trainCorrect = calculateAccuracy(train_loader)
        valCorrect = calculateAccuracy(val_loader)

        # update our training history
        STATS["train_loss"].append(avgTrainLoss)
        STATS["train_acc"].append(trainCorrect)
        STATS["val_loss"].append(avgValLoss)
        STATS["val_acc"].append(valCorrect)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

        # Compute and print the average accuracy fo this epoch when tested over all test images
        test_accuracy = calculateAccuracy(test_loader)

        # Accumulating test accuracies to draw the plot
        STATS["test_acc"].append(test_accuracy)
        
        # we want to save the model if the accuracy is the best 
        if test_accuracy > best_accuracy:
            saveModel() 
            best_accuracy = test_accuracy

    # finish measuring how long training took
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    # now we can calculate the training time in seconds
    print(" Training + validation time: ", str(start.elapsed_time(end)/1000.0), " seconds\n", file=OUT_FILE)

    # .info file output summary data 
    print("********************************************************************************\n\n",
          "MODEL TESTING STATS:\n",  
          file=OUT_FILE)

    # STDOUT  
    for i in range(len(STATS["test_acc"])):
        print(' For epoch', i+1,'the TEST accuracy over the whole TEST dataset is %d %%' % (STATS["test_acc"][i]),file=OUT_FILE) 
    
    # Now we can plot the training and validation loss curve... 
    # TODO WRAP-IT-ALL ON THE SAME PLOT (USING LEGENDS AND STUFF)
    plt.plot(STATS["train_loss"], color='red', label="train loss")
    plt.plot(STATS["val_loss"], color='blue', label="validation loss")
    plt.title("TRAINING LOSS CURVE")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss coeficient")
    plt.legend()
    plt.savefig(MODEL_PATH + '/model-loss-curves.png')
    plt.show()

    # ...the training & validation accuracy progression...
    plt.plot(STATS["train_acc"], color='red', label="training accuracy")
    plt.plot(STATS["val_acc"], color='blue', label="validation accuracy")
    plt.plot(STATS["test_acc"], color='purple', label="test accuracy")
    plt.title("ACCURACY PROGRESSION")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy (%)")    
    plt.legend()
    plt.savefig(MODEL_PATH + '/model-accuracies.png')
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

######################################## 5.1 - AUXILIAR FUNCTIONS #################################

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(MODEL_PATH + '/labels-prediction.png')
    plt.show()

###################################### 5.2 - MODEL TESTING FUNCTION ###############################

# Function to test the model with a batch of images and show the labels predictions
def testBatch():

    NUMBER_OF_SAMPLES = 24

    # JBERICAT: Create a loader for the test subset which will read the data for the final prediction test. 
    # Note that now we want to shuffle images to get random samples of every class, so we set it to true. 
    # Also, we only need a small sample of images for this test (24 is quite enough).       
    predictions_loader = DataLoader(test_data, batch_size=NUMBER_OF_SAMPLES, shuffle=True, num_workers=0)

    # get batch of images from the test DataLoader  
    images, labels = next(iter(predictions_loader)) 

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('\nReal labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(NUMBER_OF_SAMPLES)),file=OUT_FILE)
  
    # Let's see what if the model identifies the  labels of these example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('\nPredicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(NUMBER_OF_SAMPLES)),file=OUT_FILE)

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                            6. MAIN CODE                                   ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

if __name__ == "__main__":
    
    # Let's build our model while benchmarking the GPU usage stats
    train(EPOCHS)
    print('[INFO] Finished Training')
   
    # Let's load the model we just created and test the accuracy per label
    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "/trained-model.pth"))

    # Test with batch of images
    testBatch()

    # Generating final results report tarball file
    import tarfile
    import shutil

    OUT_FILE.close()
    archive = tarfile.open(MODEL_PATH+".tar.gz", "w|gz")
    archive.add(MODEL_PATH, arcname="")
    archive.close()

    print("\n[INFO] Training model and summary file saved at: " + os.path.abspath(MODEL_PATH) + ".tar.gz\n")

    # Deleting temp folder
    shutil.rmtree(MODEL_PATH)