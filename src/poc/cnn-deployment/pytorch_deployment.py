"""
Title::

    pytorch_deployment_PoC#1_v1.0-.py 

Conda environment: "conda activate py364_clone"

    TODO Python 3.6.4 + pytorch + CUDA, etc....

Description::

    Algorithm that makes inferences over the colected data and performs classification, 
    but WITHOUT applying localization boxes (just image classification).
    
    IMPORTANT NOTICE: NO REAL-TIME PROCESSING ON THIS PoC VERSION (PoC#1_v1.0). 
    
    Image analysis is done OFFLINE once the image collection by the drone's is
    finished. This is a WORK-IN-PROGRESS. The goal is to process images on 
    realtime. 

    ############################################################################
    ##
    ##  PoC: Image classification WITHOUT adding bounding boxes to the FLIR images, 
    ##          which contain wilfire instances from one class only
    ##
    ###########################################################################

    # On this implementation (REAL CASE SIMULATION) we actually don't know what
    # the object classification is, so it must be visually checked afterwards. 
    # However, we can be sure that the accuracy of the prediction will be of the 92% 
    # with a small variation index (the model accuracy is pretty stable, 
    # as seen in the training stats on the section 6.x.x.x). THEREFORE, 
    # we don't add class labels to the images extracted from the deployment
    # dataset (labeled as unkown-class on the deployment dataset). Instead, 
    # we'll be labeling it with the frame/image sequential id (adding the filename 
    # would be ideal but there is no more time for fancy stuff). This way, an operator 
    # could verify the validity of a small sample, but big enough to be of significance.

Inputs::

    1. Images taken during the drones fly-by over the PoC area, which are already converted to FLIR

    2. Trained CNN Model .pth file -> /usr/PoC/CNN/trained-model.pth

Output::

    1. Classification results: /home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/out/


Original author::

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
    2 - https://medium.com/analytics-vidhya/guide-to-object-detection-using-pytorch-3925e29737b9 


TODO list: 
    1 - "Realtimize" the whole process: It would require the implementation of
        syncronization mechanisms (e.g. a queue to retrieve the data from "airsim_rec.txt", mutex or locks for the frame_count, etc),
    2 - Try catch clause for when the IN data folder is empty

"""
# Importing functions from the PoC Library folder /src/poc/lib
import sys
from turtle import color 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 
from poc.lib.pytorch import *

import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv 

import torch
from torch.utils.data import DataLoader
from torch.functional import Tensor
import torchvision
from torchvision import datasets, transforms

# Path constants
POC_FOLDER = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 
FLIR_BUFFER = POC_FOLDER + 'flir_buffer/unknown-class/'
CLASSIFICATION_DIR = POC_FOLDER + 'out/'

# Data-bond constants
DEPLOY_DATA_DIR = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer"
CNN_IMG_SIZE = 229

'''
We're going to process the images one by one to simulate the inference process made 
by a GPU device embeded into the drone's companion computer (e.g. NVIDIA Jetson Nano).
Hence, batch-sizes need to be of one
'''
BATCH_SIZE = 1

# Model-bond constants
MODEL_VERSION = 3
MODEL_PATH = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/"

# Misc. constants
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
GREEN_COLOR = [0,255,0]
YELLOW_COLOR = [0,255,255]
RED_COLOR = [0,0,255]

# Function definitions
def model_inference():

    '''TODO DOCU - Function to test the model with a batch of images and show the labels predictions'''

    # stdout
    print("\n****************************************************************************************\n\n" + 
          "   CNN Deployment: Model inference test for image classification WITHOUT localization" + '\n'
          "       (That is, we're just adding bounding boxes to identify the WHOLE images)" + '\n'
          "\n****************************************************************************************\n")

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
    deploy_data = (datasets.ImageFolder(root=DEPLOY_DATA_DIR, transform=transformations))
    
    # Create a loader for the deploy set which will read the data within batch size and put into memory. 
    deploy_loader = DataLoader(deploy_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # DEBUG - batch_size=len(deploy_data)

    # Define the class labels
    classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

    print( "[INFO] - The number of images in a deploy set is: " + str(len(deploy_data)) + "\n" )
    print( "[INFO] - The batch-size is: " + str(BATCH_SIZE) + '\n')
    
    #######################################################################
    #                STEP 2: Loading the CNN model and data  
    #######################################################################

    # Let's load the model that got best accuracy during training (86%) for 
    # the PoC-ONE dataset:

    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))

    # Define your execution device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("[INFO] - Model deployed on", device, "device"+'\n')

    # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()

    # Convert model parameters and buffers to CPU or Cuda
    # model.to(device)

    # Preparing the data to build a table 
    print('[INFO] - Using the model to make inferences over the bulk data. This might take a minute or two...'+'\n')
 
    #######################################################################
    #         STEP 3: Peforming predictions over the imported data  
    #######################################################################

    for images,labels in deploy_loader:
    
        outputs = model(images)

        # obtain data from tensor (energy scores)
        scores = outputs.detach().numpy()[0]
        high_wildfire_score = scores[0]
        low_wildfire_score = scores[1]
        no_wildfire_score = scores[2]

        score_info = ('No-wildfire score: ' + str(no_wildfire_score) + '\n'
        + 'Low intensity wildfire score:' + str(low_wildfire_score) + '\n'
        + 'High intensity wildfire score:' + str(high_wildfire_score) + '\n')

        # set "human-readable" prediction 
        _, predicted = torch.max(outputs, 1)
        predicted_label = classes[predicted]

        # Show / save prediction image - TODO - def function + NO GRID!
        img_grid = torchvision.utils.make_grid(images)
        img_grid = img_grid / 2 + 0.5     # unnormalize
        npimg = img_grid.numpy()

        fig = plt.figure(figsize=(5,5), facecolor="red")

        #plt.subplot(211)
        plt.xticks([])
        plt.yticks([])
        plt.title("Prediction: " + predicted_label)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.subplots(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
        ## I changed the fig size to something obviously big to make sure it work
        #plt.tight_layout()
        #plt.subplot(212)
        plt.text(114,252, score_info, ha="center", va="center", fontsize=12, bbox={"facecolor":"white", "alpha":1})

        plt.show()
 
    # Archive predictions on the out dir
    flir_out = str(CLASSIFICATION_DIR) + str(TIMESTAMP)
    os.mkdir(flir_out)

    # Moving the flir_buffer's dir content to the output dir
    source = FLIR_BUFFER
    dest1 = flir_out
    files = os.listdir(source)
    for f in files:
        shutil.move(source+f, dest1)

    # Printing the summary
    print('[INFO] - The CNN model deployment has finished successfully. See the output .png files to visually check how accurate the predictions were.' + '\n')
    print('[INFO] - OUTPUT FILES: ' + '\n\n' + 
          '               - Classification results: ' + str(flir_out) + '\n\n')

# Main code
if __name__ == '__main__':

    model_inference()