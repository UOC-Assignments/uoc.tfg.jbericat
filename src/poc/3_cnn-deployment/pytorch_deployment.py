'''
Title::

    pytorch_deployment.py 

Conda environment: 'conda activate py364_clone'

    TODO Python 3.6.4 + pytorch + CUDA, etc....

Description::

    Algorithm that makes inferences over the colected data and performs classification, 
    but WITHOUT applying localization boxes (just image classification).
       
    Image analysis is done OFFLINE once the image collection by the drone's is
    finishe (due hardware limitations). However, on this second version of the 
    algorythm, input data is serialized. 

    Model inference test for image classification WITHOUT localization' + '\n'
          '       (That is, we're just adding bounding boxes to identify the WHOLE images)' + '\n'


Inputs::

    1. Trained CNN Model .pth file:
        
        -> /usr/poc/2_cnn-training/trained-model.pth

    2. Images taken during the drones fly-by over the PoC area, 
    which are already converted to FLIR: 
       
        -> /usr/poc/3_cnn-deployment/flir_buffer/unknown-class/

Output::

    1. Classification results:
    
        -> /usr/poc/3_cnn-deployment/out/%TIMESTAMP%/

    DETAILS:

    # On this implementation (REAL CASE SIMULATION) we actually don't know what
    # the object classification is BEFORE being processed by the model (they're unlabeled 
    # images), so images must be visually checked afterwards.
    # However, we can be sure that the accuracy of the predictions will be of the 92% 
    # with a small variation index. THEREFORE, we don't add class labels to the 
    # images extracted from the deployment dataset (labeled as unkown-class). Instead, 
    # we'll do the labeling thing just adding a frame/image sequential id on the output 
    # images filename. 


Original author::

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
    2 - https://medium.com/analytics-vidhya/guide-to-object-detection-using-pytorch-3925e29737b9 


TODO list: 

    1 - Try catch clause to check if there is data on the flir_buffer data folder (is empty)


'''
# Importing functions from the PoC Library folder /src/poc/lib
import sys
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 
from poc.lib.pytorch import *

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

# Path constants
GIT_FOLDER = '/home/jbericat/Workspaces/uoc.tfg.jbericat'
BASE_FOLDER = GIT_FOLDER + '/usr/poc'
IN_DIR = BASE_FOLDER +'/3_cnn-deployment/flir-buffer'
OUT_DIR = BASE_FOLDER + '/3_cnn-deployment/out'

# Data-bond constants
'''
We're going to process the images one by one to simulate the inference process made 
by a GPU device embeded into the drone's companion computer (e.g. NVIDIA Jetson Nano).
Hence, batch-sizes need to be of one
'''
BATCH_SIZE = 1
CNN_IMG_SIZE = 229

# Model-bond constants
MODEL_VERSION = 3
MODEL_PATH = BASE_FOLDER + '/2_cnn-training'

# Misc. constants
TIMESTAMP = time.strftime('%Y%m%d-%H%M%S')

# Function definitions
def model_inference():

    '''TODO DOCU - Function to test the model with a batch of images and show the labels predictions'''

    # STDOUT info
    print('\n****************************************************************************************\n\n' + 
          '   3. CNN Deployment Algorythm '+
          '\n****************************************************************************************\n')

    #######################################################################
    #               STEP 1: Preparing the deployment data 
    #######################################################################

    # Define transformations for the training and test subsets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL ONLY (GRAYSCALE))
        transforms.Resize(CNN_IMG_SIZE)
    ])

    # Create an instance for deployment
    deploy_data = (datasets.ImageFolder(root=IN_DIR, transform=transformations))
    
    # Create a loader for the deploy set which will read the data within batch size and put into memory. 
    deploy_loader = DataLoader(deploy_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1) # DEBUG - num_workers one CPU, so we won't accidentally shuffle the deploy images (they are ordered!). This can happens if two different images are processed by two different threads 

    # Define the class labels
    classes = ('high-intensity-wildfire', 'low-intensity-wildfire', 'no-wildfire')

    # STDOUT Info
    print( '[INFO] - The number of images in a deploy set is: ' + str(len(deploy_data)) + '\n' )
    print( '[INFO] - The batch-size is: ' + str(BATCH_SIZE) + '\n')
    
    #######################################################################
    #                STEP 2: Loading the CNN model and data  
    #######################################################################

    # Let's load the model that got best accuracy during training (86%) for 
    # the PoC-ONE dataset:

    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + '/trained-model.pth'))

    # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()

    # STDOUT Info
    print('[INFO] - Using the model on CPU device to make inferences over the serialized data. This might take a minute or two...'+'\n')
 
    #######################################################################
    #         STEP 3: Peforming predictions over the imported data  
    #######################################################################

    # Archive predictions on the out dir
    flir_out = str(OUT_DIR) + '/' +  str(TIMESTAMP)
    os.mkdir(flir_out)

    frame_id = 0

    for images,labels in deploy_loader:
    
        outputs = model(images)

        # obtain data from tensor (energy scores)
        scores = outputs.detach().numpy()[0]
        high_wildfire_score = scores[0]
        low_wildfire_score = scores[1]
        no_wildfire_score = scores[2]

        # plot text formatting
        score_info = ('No-wildfire score: ' + str(no_wildfire_score) + '\n'
        + 'Low-intensity wildfire score:' + str(low_wildfire_score) + '\n'
        + 'High-intensity wildfire score:' + str(high_wildfire_score) + '\n')

        # showing prediction in a more 'human-readable' way
        _, predicted = torch.max(outputs, 1)
        predicted_label = classes[predicted]

        # Show / save prediction image
        img_grid = torchvision.utils.make_grid(images)
        img_grid = img_grid / 2 + 0.5     # unnormalize
        npimg = img_grid.numpy()

        # PNG summary to OUT DIR
        if predicted_label == 'no-wildfire':
            fig = plt.figure(figsize=(5,5), facecolor='green')
        if predicted_label == 'low-intensity-wildfire':
            fig = plt.figure(figsize=(5,5), facecolor='yellow')
        if predicted_label == 'high-intensity-wildfire':
            fig = plt.figure(figsize=(5,5), facecolor='red')
        plt.xticks([])
        plt.yticks([])
        plt.title('Prediction: ' + predicted_label, bbox={'facecolor':'white', 'alpha':1})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.text(114,252, score_info, ha='center', va='center', fontsize=12, bbox={'facecolor':'white', 'alpha':1})
        plt.savefig(flir_out + '/frame-' + str(frame_id) + '.png')
        plt.close(fig)

        # Frame counter
        frame_id += 1

    # TODO - Delete flir_buffer dir content
    

    # STDOUT Info
    print('[INFO] - The CNN model deployment has finished successfully. See the output .png files to visually check how accurate the predictions were.' + '\n')
    print('[INFO] - OUTPUT FILES: ' + '\n\n' + 
          '               - Classification results: ' + str(flir_out) + '\n\n')

# Main code
if __name__ == '__main__':

    model_inference()