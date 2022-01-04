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

Inputs::

    1. Images taken during the drones fly-by over the PoC area, which are already converted to FLIR

    2. Trained CNN Model .pth file -> /usr/PoC/CNN/trained-model.pth

Output::
    1. Inference summary: /home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/out/inference-predictions.log

    2. Classification results: /home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer/


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

# IMPORTS

## PART I 

import cv2 as cv # https://stackoverflow.com/questions/50909569/unable-to-import-cv2-module-python-3-6
import os

from tabulate import tabulate

## PART II 

# DEBUG: We copied some code (defs) into this project folder in a rush. Create a shared lib in /usr/lib by instance....

# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 

from poc.lib.CNN_models import *

import numpy as np
import sys
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.functional import Tensor
import matplotlib.pyplot as plt

# Base folders
POC_FOLDER = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 
FLIR_BUFFER = POC_FOLDER + 'flir_buffer/'
PREDICTIONS_SUMMARY = open(os.path.abspath(POC_FOLDER + 'out/frame-predictions.log'), "w")   

#UE4_ZONE_6 = 6
#UE4_ZONE_7 = 7

GREEN_COLOR = [0,255,0]
YELLOW_COLOR = [0,255,255]
RED_COLOR = [0,0,255]

#data-bond constants
DEPLOY_DATA_DIR = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer"
CNN_IMG_SIZE = 229

#model-bond constants
MODEL_VERSION = 3
MODEL_PATH = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/"

# To simulate a GPU device embeded into the drone's companion computer (e.g. NVIDIA Jetson Nano)
# now we're going to process the images one by one. So we won't be using batch sizes
BATCH_SIZE = 128

def add_bounding_box(prediction):
    #TODO DOC - https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    borderType = cv.BORDER_CONSTANT

    # Load an image
    # TODO - Path structure is a bit of a mess...
    frame_path = FLIR_BUFFER + 'unknown-class/frame-' + str(prediction[0]) + '.png'
    src = cv.imread(cv.samples.findFile(frame_path), cv.IMREAD_COLOR)

    # Check if the image was correctly loaded 
    if src is None:
        print ('Error opening image!')
        return -1

    
    #TODO DOC
    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * src.shape[1])  # shape[1] = cols
    right = left

    if prediction[1] == 'no-wildfires':
        #add_green_boundingbox
        color = GREEN_COLOR

    elif prediction[1] == 'low-intensity-wildfires':
        #add_yellow_boundingbox
        color = YELLOW_COLOR

    elif prediction[1] == 'high-intensity-wildfires':
        #add_green_boundingbox
        color = RED_COLOR

    #TODO DOC
    dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, color)
   
    # SAVE TO FILE (OVERWRITING THE BUFFER IS OK)
    cv.imwrite(frame_path, dst)


# PART II DEFINITIONS

def model_inference():

    '''TODO DOCU - Function to test the model with a batch of images and show the labels predictions'''

    # stdout
    print("\n****************************************************************************************\n\n" + 
          "   CNN Deployment: Model inference test for image classification WITHOUT localization" + '\n'
          "       (That is, we're just adding bounding boxes to identify the WHOLE images)" + '\n'
          "\n****************************************************************************************\n")

    #######################################################################
    # STEP 1: Loading the deployment data 
    #######################################################################

    # Define transformations for the training and test subsets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        # Normalizing the images ___________
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL ONLY (GRAYSCALE)
        # We need square images to feed the model (the raw dataset has 640x512 size images)
        # DEBUG - UNCOMMENT NEXT LINE FOR v4 and v9 DATASETS
        #transforms.RandomResizedCrop(512),
        # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
        transforms.Resize(CNN_IMG_SIZE)
    ])

    # Create an instance for deployment
    deploy_data = (datasets.ImageFolder(root=DEPLOY_DATA_DIR, transform=transformations))
    
    # Create a loader for the test set which will read the data within batch size and put into memory. 
    # Note that each shuffle is set to false, since we will be creating a frame-by-frame animation 
    deploy_loader = DataLoader(deploy_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # DEBUG - batch_size=len(deploy_data)


    print( "[INFO] - The number of images in a deploy set is: " + str(len(deploy_data)) + "\n" )
    print( "[INFO] - The batch-size is: " + str(BATCH_SIZE) + '\n')
    
    #######################################################################
    # STEP 2: Loading the CNN model and importing the data  
    #######################################################################

    # Let's load the model that got best accuracy during training (86%) for 
    # the PoC-ONE dataset:

    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] - Model deployed on", device, "device"+'\n')

     # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    
    # DEBUG - THE LOOP STARTS HERE
    # for j in range(len(deploy_data)):
    mydata = []
    frame_id = 0 #COUNTER


    # Preparing the data to build a table 
    print('[INFO] - Using the model to make inferences over the bulk data. This might take a minute or two...'+'\n')

    for i, (images, labels) in enumerate(deploy_loader, 0):
    
        # get batch of images from the test DataLoader  
        #images, labels = next(iter(deploy_loader)) 

        # MOVING DATA TO THE GPU MEM SPACE
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        #######################################################################
        # STEP 3: Peforming predictions over the imported data  
        #######################################################################
        
        # Let's see what if the model identifies the labels of these example
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        #######################################################################
        # STEP 4: Showing results  
        #######################################################################

        classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

        for j in range(len(outputs)):
            # On this implementation (REAL CASE SIMULATION) we actually don't know what
            # the object classification is, so it must be visually checked afterwards. 
            # However, we can be sure that the accuracy of the prediction will be of the 86% 
            # with a small variation index (the model accuracy is pretty stable, 
            # as seen in the training stats on the section 6.x.x.x). THEREFORE, 
            # we don't add class labels to the images extracted from the deployment
            # dataset (labeled as unkown-class on the deployment dataset). Instead, 
            # we'll be labeling it with the frame/image sequential id (adding the filename 
            # would be ideal but there is no more time for fancy stuff). This way, an operator 
            # could verify the validity of a small sample, but big enough to be of significance.

            # assign data
            predicted_label = classes[predicted[j]]
            mydata.append([frame_id, predicted_label])
            frame_id += 1
    
    # create header
    head = ["Frame ID", "Classification Result"]
    
    # print the table to info file
    print(tabulate(mydata, headers=head, tablefmt="grid"), file=PREDICTIONS_SUMMARY)

    # At last, we can add colored boundinb boxes to each image (neither with 
    # localization nor with object detection, just plain classification)
    # Bounding colors:
    # - Green = no-wildfires class
    # - Yellow = low-intensity-wildfires class
    # - Red = high-intensity-wildfires class

    for i in range(len(deploy_data)): # TODO DEBUG - THIS IS A PATCH! We're creating a myData list of size multiple of the batch-size, instead of the deploy data size
        # TODO - DOC
        add_bounding_box(mydata[i])
    print('[INFO] - The CNN model deployment has finished successfully. See the output .png files to visually check how accurate the predictions were.' + '\n')
    print('[INFO] - OUTPUT FILES: ' + '\n\n' + 
          '               - Inference summary: ' + str(POC_FOLDER + 'out/inference-predictions.log') + '\n\n' +
          '               - Classification results: ' + str(FLIR_BUFFER) + '\n\n')

import torch.onnx 
import torch
from torchvision import transforms
 
# CALLING MAIN.
if __name__ == '__main__':


    ############################################################################
    ##
    ##  PoC: Image classification WITHOUT adding bounding boxes to the FLIR images, 
    ##          which contain wilfire instances from one class only
    ##
    ###########################################################################

    model_inference()

