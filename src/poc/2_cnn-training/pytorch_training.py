"""
Title::

    pytorch_training.py 

Python environment: 

    TODO Python 3.4.6 + pytorch x.y....... 

Description::

    this program sets the appropiate envionment hyper-parameters and then loads the
    selected CNN model (implemented on the CNN_models.py file) in order to train it using
    the synthetic data (thermal / night-vision simulated wildfire images) previosly created 
    using the Unreal Engine editor. Once the model gets trained, a summary file is created
    including the trained CNN architecure blueprint and other performance data.

Inputs::

    TODO - Model & Dataset 

Output::

    Tarball file "/usr/poc/2_cnn-training/cnn-training_%TIMESTAMP%.tar.gz" including:

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
    4 - https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    5 - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html  
    6 - https://discuss.pytorch.org/t/plotting-loss-curve/42632/2

TODO list: 

    1 - Extract function to definitions file, so they can be shared (CNN_models.py). 
    2 - Separate definitions from implementations. No time though....


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

from torch.functional import Tensor

# This script's execution timestamp
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")


# STDOUT info
print("\n****************************************************************************************\n\n" + 
        "                                  2. CNN Training Algorythm" + '\n'
        "\n****************************************************************************************\n")

# CNN model & Dataset versions (we do not waste time implementing error control)
print("Set the model version you want to train:" + "\n\n" + 
                        " 1 = v1.0 -> 3 layers CNN (simplified LeNet)"+ "\n" +
                        " 2 = v2.0 -> 5 layers CNN (LeNet)" + "\n" +
                        " 3 = v3.0 -> 14 layers CNN (custom)"+ "\n")

MODEL_VERSION = input("Please choose an option (1-3 - Default = 3): ") or '3'

print("Set the dataset version you want to use to train the model:"+"\n\n"+ 
                            " 4 = v4.0 -> ARCHIVED (EXPERIMENTAL)"+"\n" + 
                            " 5 = v5.0 -> ARCHIVED (EXPERIMENTAL)"+"\n" +
                            " 6 = v6.0 -> ARCHIVED (EXPERIMENTAL)"+"\n" +
                            " 7 = v7.0 -> 600 samples: 3 classes, close distance images (SMALLEST DATASET, appropiate for adjusting parameters)"+"\n"+
                            " 8 = v8.0 -> 1.5K samples: 4 classes, close distance images"+"\n"+
                            " 9 = v9.0 -> 1.5K samples: 3 classes, close distance images"+"\n"+
                            " 10 = v10.0 -> ARCHIVED (EXPERIMENTAL)"+"\n" +
                            " 11 = v11.0 -> ARCHIVED (EXPERIMENTAL)"+"\n" +                            
                            " 12 = v12.0 -> 3.7K samples: 3 classes, close, mid & long distance images (Improved no-wildfire class images)."+"\n")

DATASET_VERSION = input("Please choose an option (4-12 - Default = 7): ") or '7'

GIT_DIR = '/home/jbericat/Workspaces/uoc.tfg.jbericat'

# model storage path
OUT_DIR = GIT_DIR + "/usr/poc/2_cnn-training/cnn-training_" + str(TIMESTAMP)

# Dataset paths - DEBUG: replace abs for rel paths?
ROOT_DATA_DIR = '/usr/poc/1_datasets/curated-data'
TRAIN_DATA_DIR = GIT_DIR + ROOT_DATA_DIR + '/v' + DATASET_VERSION + '.0/training+validation'
TEST_DATA_DIR = GIT_DIR + ROOT_DATA_DIR + '/v' + DATASET_VERSION + '.0/test'

# opening / creating the file where to store the results
os.mkdir(OUT_DIR)
OUT_FILE = open(OUT_DIR + "/trained-model.info", "w")

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
EPOCHS = 20

# The PoC's dataset consists of 500x2=1000 training images and 200x2=400 test images (we're adding the augmented dataset). 
# Hence, we define a batch size of X to load YY & ZZ batches of images respectively on each epoch:
BATCH_SIZE = 128

# Learning rate: 
LEARNING_RATE = 0.0001

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
    # Now we just resize into any of the common input layer sizes (32??32, 64??64, 96??96, 224??224, 227??227, and 229??229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

# Define augmentation techniques (mirroring) to increase the number of samples / images on the training and test subsets
mirroring_augmentations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL (GRAYSCALE)
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation technique: Horizontal Mirroring
    # Now we just resize into any of the common input layer sizes (32??32, 64??64, 96??96, 224??224, 227??227, and 229??229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

rotation_augmentations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL (GRAYSCALE)
    transforms.RandomRotation(45), # Augmentation technique: Image rotation
    # Now we just resize into any of the common input layer sizes (32??32, 64??64, 96??96, 224??224, 227??227, and 229??229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

shear1_augmentations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL (GRAYSCALE)
    transforms.RandomAffine(45), # Augmentation technique: Image shifting (shear)
    # Now we just resize into any of the common input layer sizes (32??32, 64??64, 96??96, 224??224, 227??227, and 229??229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

shear2_augmentations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL (GRAYSCALE)
    transforms.RandomAffine(75), # Augmentation technique: Image shifting (shear)
    # Now we just resize into any of the common input layer sizes (32??32, 64??64, 96??96, 224??224, 227??227, and 229??229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

########################### 2.2 - LOADING TRAINING, VALIDATION & TEST DATA #######################

# Create an instance for training. 
train_data = (datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transformations) + 
             datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=mirroring_augmentations) + 
             datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=rotation_augmentations) +
             datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=shear1_augmentations) +
             datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=shear2_augmentations))

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
test_data = (datasets.ImageFolder(root=TEST_DATA_DIR, transform=transformations))

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# calculate steps per epoch for training and validation set (needed to track training + validation loss & accuracy)
trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(val_loader.dataset) // BATCH_SIZE

# initialize a dictionary to store training history
STATS = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "test_acc": [],
    "final_test_acc": []
}

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


# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 
from poc.lib.pytorch import *

# DEBUG: The next function has been added to the CNN_models library, it might be removed from here safely....

def set_model_version(input):
    if (input == '1'):
        selection = Network_v1()
        
    elif (input == '2'):
        selection = Network_v2()

    elif (input == '3'): 
        selection = Network_v3()

    return selection

# END OF DEBUG

# Instantiate the selected neural network model class imported from the src/CNN/CNN_Models.py file
# DEBUG: This line is already on the main function, IDK what's doing here....
model = set_model_version(MODEL_VERSION)
# END OF DEBUG

# .info file output summary data 
print("\n********************************************************************************\n\n",
      "CNN BLUEPRINT:\n\n ",  
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
    torch.save(model.state_dict(), OUT_DIR + "/trained-model.pth")

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
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n********************************************************************************\n\n",
          "MODEL TRAINING STATS:\n\n", "- Model trained on", device, "device",
          file=OUT_FILE)

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # STDOUT INFO
    print("[INFO] Training the convolution neural network model, hold-on tight...\n")

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
        
        #################### 5.2.2 - MODEL VALIDATION ####################

        # DEBUG_  validation might better be done after EACH ITERATION over the training set OF THE SAME EPOCH, 
        #         but we're doing it after the ALL THE ITEARATIONS over the training set OF THE SAME EPOCH. 
        #         documentation must ve reviewed...

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

	#################### 5.2.3 - TRAIN & VAL STATS ####################
    
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
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
    print(" - Training time: ", str(start.elapsed_time(end)/1000.0), " seconds\n", file=OUT_FILE)

    # .info file output summary data 
    print("********************************************************************************\n\n",
          "MODEL TESTING STATS:\n",  
          file=OUT_FILE)

    # STDOUT  
    for i in range(len(STATS["test_acc"])):
        print('  - For epoch', i+1,'the TEST accuracy over the whole TEST dataset is %d %%' % (STATS["test_acc"][i]),file=OUT_FILE) 
    
    # Now we can plot the training and validation loss curve... 
    fig=plt.figure()
    plt.plot(STATS["train_loss"], color='red', label="train loss")
    plt.plot(STATS["val_loss"], color='blue', label="validation loss")
    plt.title("TRAINING LOSS CURVE")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss coeficient")
    plt.legend()
    plt.savefig(OUT_DIR + '/model-loss-curves.png')
    plt.close(fig)

    # ...& the learning curve.
    fig=plt.figure()
    plt.plot(STATS["train_acc"], color='red', label="training accuracy")
    plt.plot(STATS["val_acc"], color='blue', label="validation accuracy")
    plt.plot(STATS["test_acc"], color='purple', label="test accuracy")
    plt.title("LEARNING CURVE")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy (%)")    
    plt.legend()
    plt.savefig(OUT_DIR + '/model-accuracies.png')
    plt.close(fig)

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
from tabulate import tabulate

###################################### 6.1 - MODEL TESTING FUNCTION ###############################

# Function to test the model with a batch of images and show the labels predictions
def testBatch():

    # stdout
    print("\n********************************************************************************\n\n" + 
          " FINAL PREDICTION TEST:\n", 
          file=OUT_FILE)

    NUMBER_OF_SAMPLES = 24

    # Create a loader for the test subset which will read the data for the final prediction test. 
    # Note that now we want to shuffle images to get random samples of every class, so we set it to true. 
    # Also, we only need a small sample of images for this test (24 is quite enough).       
    predictions_loader = DataLoader(test_data, batch_size=NUMBER_OF_SAMPLES, shuffle=True, num_workers=0)

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - Model tested on", device, "device",
          file=OUT_FILE)

    # This directive MUST be enabled for testing
    #  (I'd say it wasn't present on the original code)
    model.eval()

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # get batch of images from the test DataLoader  
    images, labels = next(iter(predictions_loader)) 
    
    # MOVING DATA TO THE GPU MEM SPACE
    images = Variable(images.to(device))
    labels = Variable(labels.to(device))

    # show all images as one image grid
    img_grid = Tensor.cpu(torchvision.utils.make_grid(images))
    img_grid = img_grid / 2 + 0.5     # unnormalize
    npimg = img_grid.numpy()
    fig=plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(OUT_DIR + '/labels-prediction.png')
    plt.close(fig)
    
    # Let's see what if the model identifies the labels of these example
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Showing the accuracy of the final test
    STATS["final_test_acc"].append(calculateAccuracy(predictions_loader))
    print(" - Final test accuracy: %d %%" % (STATS["final_test_acc"][0])  ,"\n", file=OUT_FILE)

    # Preparing the data to build a table 
    mydata = []
    for j in range(NUMBER_OF_SAMPLES):

        out = Tensor.cpu(outputs[j])
        scores = out.detach().numpy()

        # assign data
        real_label = str(classes[labels[j]])
        predicted_label = classes[predicted[j]]
        mydata.append([real_label, predicted_label, scores])
    
    # create header
    head = ["Real Label", "Predicted Label", "Output score [High / Low / No]"]
    
    # print the table to info file
    print(tabulate(mydata, headers=head, tablefmt="grid"), file=OUT_FILE)

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
    print('[INFO] Finished Training. Generating summary tarball...')
   
    # Let's load the model we just created and test the accuracy per label
    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(OUT_DIR + "/trained-model.pth"))

    # Test with batch of images
    testBatch()

    # Generating final results report tarball file
    import tarfile
    import shutil

    OUT_FILE.close()
    archive = tarfile.open(OUT_DIR +".tar.gz", "w|gz")
    archive.add(OUT_DIR, arcname="")
    archive.close()

    print("\n[INFO] Training model and summary file saved at: " + OUT_DIR + ".tar.gz\n")

    # Deleting temp folder
    shutil.rmtree(OUT_DIR)