# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model


# PARAMETRIZATION

import os

DATASET_IMG_SIZE = 229
TRAIN_DATA_DIR = os.path.abspath('src/CNN/data/training+validation/')  
TEST_DATA_DIR = os.path.abspath('src/CNN/data/test/')  
IMG_CHANNELS = 3

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                       1. IMPORTING THE DATA                               ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

#https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1.1 - Loading and normalizing the data.
# Define transformations for the training and test sets

transformations = transforms.Compose([
    transforms.ToTensor(),
    # Normalizing the images ___________
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # We need square images to feed the model (the raw dataset has 640x512 size images)
    transforms.RandomResizedCrop(512),
    # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
    transforms.Resize(DATASET_IMG_SIZE)
    ])

# 1.2 - Create an instance for training. 
train_data = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transformations)

# the PoC's dataset consists of 608 training images and 204 test images. 
# We define the batch size of X to load YY & ZZ batches of images respectively:
batch_size = 10

# The number of labels correspond to the amount of classes we defined on previous
# stages of this project. To sum-up, we have: 
# 
# - Class #1: High-intensity wildfires 
# - Class #2: Medium-intensity wildfires 
# - Class #3: Low-intensity wildfires
# - Class #4: Images with no wildfires at all
number_of_labels = 4 

# 1.3 - Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

# 1.4 - Create an instance for testing
test_data = datasets.ImageFolder(root=TEST_DATA_DIR, transform=transformations)

# 1.5 - Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('high-intensity-wildfire', 'medium-intensity-wildfire', 'low-intensity-wildfire ', 'no-wildfire')


###################################################################################################
###################################################################################################
####                                                                                           ####   
####                                       2. TRAINING THE MODEL                               ####
####                                                                                           ####   
###################################################################################################
###################################################################################################


# 2.1 - Define a convolution neural network  
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Network(nn.Module):
     
    def __init__(self):
        super(Network, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=IMG_CHANNELS, out_channels=IMG_CHANNELS*4, kernel_size=11, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(IMG_CHANNELS*4)
        self.conv2 = nn.Conv2d(in_channels=IMG_CHANNELS*4, out_channels=IMG_CHANNELS*4, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(IMG_CHANNELS*4)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=IMG_CHANNELS*4, out_channels=IMG_CHANNELS*8, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(IMG_CHANNELS*8)
        self.conv5 = nn.Conv2d(in_channels=IMG_CHANNELS*8, out_channels=IMG_CHANNELS*8, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(IMG_CHANNELS*8)
        self.fc1 = nn.Linear(264600, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 264600)
        output = self.fc1(output)

        return output

# Instantiate a neural network model 
model = Network()

# 2.2 - Define a loss function
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 2.3 - Train the model on the training data
from torch.autograd import Variable

# Function to save the model
def saveModel():
    path = "./bin/CNN_Model_batch-size_" + str(batch_size) + ".pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader: # BUGFIX #126 ->  See TASK05.6: https://github.com/UOC-Assignments/uoc.tfg.jbericat/issues/96
            images, labels = images.cuda(), labels.cuda() # # BUGFIX #126 ->  See TASK05.6: https://github.com/UOC-Assignments/uoc.tfg.jbericat/issues/96 
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

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)


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
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# 2.4:  Test the model on the test data
import matplotlib.pyplot as plt
import numpy as np

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))


# 2.5:  Adding the main code  
if __name__ == "__main__":
    
    # Let's build our model
    train(5)
    print('Finished Training')

    # Test which classes performed well
    # DEBUG - UNCOMMENT NEXT LINE! 
    #testModelAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "./bin/CNN_Model_batch-size_" + str(batch_size) + ".pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()