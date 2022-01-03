"""
Title::

    pytorch_deployment_PoC#1_v1.0-.py 

Python environment: 

    TODO Python 3.6.4 + pytorch + CUDA, etc....

Description::

    Deploying the classification algorithm (model cnn-training_v3.pth) WITHOUT
    localization boxes (just image classification).
    
    IMPORTANT NOTICE: NO REAL-TIME PROCESSING ON THIS PoC VERSION (PoC#1_v1.0). 
    
    Image analysis is done OFFLINE once the image collection by the drone's is
    finished. This is a WORK-IN-PROGRESS. The goal is to process images on 
    realtime. However, to do so I should find a way to do the create_flir_img()
    function computation in a way more efficient way. By instance, using cuda 
    would allow me to scan the 512 rows of every SEGMENT image AT ONCE (GPU has
    896 cores, so it should suffice). Another way would be just cpu-threading 
    the sequential process (IDK how efficiently I can do that with python, 
    though). 

Inputs::

    1. RAW .png images (SEGMENT-IR & SCENE-RGB) taken by the drone during the 
       surveillance flight (drone_patrol.py) -> /usr/PoC/in/%sample_folder%/

    2. Trained CNN Model .pth file -> /usr/PoC/CNN/trained-model.pth

Output::

    1. Unlabelel Image-grid summary itended for visual inspection 
       -> /usr/PoC/out/labels-prediction.png

    2. STODUT redirected to file (table with predictions ordered by frame number) 
       -> /usr/PoC/out/frame-predictions.png 


Original author::

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#MultirotorClient.moveOnPathAsync 
    2 - https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo 
    3 - TODO


TODO list: 
    1 - Paralelize create_flir_img()
    2 - "Realtimize" the whole process: It would require the implementation of
        syncronization mechanisms (e.g. a queue to retrieve the data from "airsim_rec.txt")
    3 - Try catch clause for when the IN data folder is empty

"""

# IMPORTS

## PART I 

from genericpath import isfile
from posixpath import join
import cv2 # https://stackoverflow.com/questions/50909569/unable-to-import-cv2-module-python-3-6
import os
import time
import glob
import shutil
from numpy.core.fromnumeric import take

from tabulate import tabulate
from torch.functional import Tensor, meshgrid
from torch.utils.data.sampler import BatchSampler
import torchvision

## PART II 

# DEBUG: We copied some code (defs) into this project folder in a rush. Create a shared lib in /usr/lib by instance....

from CNN_models import *

import numpy as np
from numpy.lib import utils


#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
import sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# CONSTANTS AND LITERALS - TODO - ALL THIS STUFF SHOULD BE ON A .H KINDOFF FILE OR WHATEVER BETTER STRUCTURED THAN THIS... (MODULES, PACKAGES, ETC)

# Base folders
POC_FOLDER = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 
FLIR_BUFFER = POC_FOLDER + 'flir_buffer/'
PREDICTIONS_OUT_FILE = open(os.path.abspath(POC_FOLDER + 'out/frame-predictions.log'), "w")   


# PART I

UE4_ZONE_6 = 6
UE4_ZONE_7 = 7

GREEN_COLOR = [0,255,0]
YELLOW_COLOR = [0,255,255]
RED_COLOR = [0,0,255]

# PART II

#data-bond constants
DEPLOY_DATA_DIR = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer"
CNN_IMG_SIZE = 229

# Setting a reasonable BATCH SIZE is requirede considering that:
#  
# - In the development station we have a fairly high amount of mem avaiable 
#   (4Gb RAM on my NVIDIA GTX 1650) to start with. HOWEVER, if the CNN model i
#   is going to be deployed on EMBEDDED GPU DEVICES (such as the Jetson Nano)

# - On the DEV STATION, we're rendering the AirSim simulation and evaluating 
#   the images against the neural network AT ONCE. The point here is that the
#   Unreal Engine "eats" almosthalf of the memory. On the other hand, pytorch 
#   with CUDA enabled eats the other half, so we end up having more or less 
#   200Mb of free GPU mem to load the data (almost 400Mb on images) as well 
#   as the trained model (1.5Mb only).
#
# To sum-up: for avoiding running-out of GPU memory we're gonna have to run 
# -> ( len(deploy_ata) / BATCH_SIZE ) iterations
BATCH_SIZE = 128

#model-bond constants
MODEL_VERSION = 3
MODEL_PATH = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/"

# FUNCTION DEFINITIONS

# PART I DEFINITIONS

def get_newest_item(path):
    '''
    Function that return the newest item in a base path
    Sources -> https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder 
    '''
    list_of_files = glob.glob(path+'*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# TODO DOC 
framecount = 0

def create_flir_img_v2(thermal_img_path, rgb_img_path, composite_img_path, ue4_zone):
    '''
    title::
        create_flir_img

    description::
        Thermal + RGB composite image: 'injects' the heat pixels (whites) detected by 
        IR-Thermal into the scene image to simulate a FLIR based thermal vision camera device

    inputs::
        ir
            Path to the image contaning the IR captured pixels (Thermal grayscale)
        scene
            Path to the image contaning the scene captured pixels (RGB)

    output::
        flir
            Thermal+RGB composite image (file) sequentially labeled by frame number

    author::
        Jordi Bericat Ruz - Universitat Oberta de Catalunya

    references::
        https://www.geeksforgeeks.org/how-to-manipulate-the-pixel-values-of-an-image-using-python/

    TODO::
        - CUDA paralelization of the "image fusion" (this function is slowing down the realtime thing). For instance, we could use 512 GPU cores to process all the grayscale_image matrix rows at once
        - Create a class definition and then import this export (implemented as a method) both to "create_ir_segmentation.py" and this file itself.
    '''
    # Import an image from directory:
    thermal_image = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)

    # Convert RGB image to grayscale -> https://stackoverflow.com/questions/48190894/how-to-convert-rgb-images-dataset-to-single-channel-grayscale
     
    # When using opencv, we load images into the BGR color space. Therefore, we convert BGR -> GRAY instead of RGB -> GRAY
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) 
    
    # On this version we're going to try a more efficient approach using the numpy framework for python:
    # https://www.geeksforgeeks.org/searching-in-a-numpy-array/ 

    #  looking for value 255 in arr and storing its index in i
    myTuple = np.where(thermal_image == 255)
    if ( len(myTuple[0]) ): 
        for index in range(len(myTuple[0])):
            x = myTuple[0][index]
            y = myTuple[1][index]
            grayscale_image[x, y] = 255

    # We have to explicitly tell that we want to use the global namespace to 
    # keep track of frame number we want to append to the out file
    global framecount
    cv2.imwrite(composite_img_path+'/FLIR_frame-' + str(framecount) + '.png', grayscale_image)
    framecount += 1 # NO CONCURRENCY = NO NEED TO MUTEX OR LOCKS, BUT THAT COULD CHANGE IN FUTURE VERSIONS

def thermal_vision_simulation():
    
    ### TODO -> DOCUMENTATION

    ########################################################################### 
    # STEP 1: Setting path variables
    ###########################################################################
   
    # OUTPUT to FLIR conversion buffer
    output = FLIR_BUFFER + 'unknown-class/'

    # INPUT sample folder
    sample_folder = get_newest_item(POC_FOLDER + 'in/') + '/'

    ###########################################################################
    # STEP 2: Extract SEGMENT & RGB Filenames from the airsim_rec.txt log
    ###########################################################################

    # Deleting the first line of the file (Header)
    with open(sample_folder + 'airsim_rec.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open(sample_folder + 'airsim_rec.txt', 'w') as fout:
        fout.writelines(data[1:])

    ### TODO -> DOCUMENTATION

    while ( os.stat(sample_folder + 'airsim_rec.txt').st_size != 0 ):

        # Retrieving oldest entry on the log
        with open(sample_folder + 'airsim_rec.txt') as f:
            log_entry = f.readline().strip()

        # Obtaining the SEGMENT & RGB images filenames
        log_entry_data = log_entry.rsplit('\t')
        log_entry_filenames = log_entry_data[-1].rsplit(';')
        myRGB_file = log_entry_filenames[0]
        mySegment_file  = log_entry_filenames[1]

        # Removing the previously read log entry (top line)
        with open(sample_folder + 'airsim_rec.txt', 'r') as fin:
            data = fin.read().splitlines(True)
        with open(sample_folder + 'airsim_rec.txt', 'w') as fout:
            fout.writelines(data[1:])

        #######################################################################
        # STEP 3: Generating the FLIR image 
        #######################################################################

        ### TODO -> DOCUMENTATION
        segment_img_path = sample_folder + 'images/' + mySegment_file
        rgb_img_path = sample_folder + 'images/' + myRGB_file

        ### TODO -> DOCUMENTATION
        create_flir_img_v2(segment_img_path, rgb_img_path, output, UE4_ZONE_6)
        
        #Deleting input files
        os.remove(segment_img_path)
        os.remove(rgb_img_path)

    #Deleting samples folder
    shutil.rmtree(sample_folder) 

def add_bounding_box(prediction):
    #TODO DOC - https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    borderType = cv2.BORDER_CONSTANT

    # Load an image
    # TODO - Path structure is a bit of a mess...
    frame_path = FLIR_BUFFER + 'unknown-class/FLIR_frame-' + str(prediction[0]) + '.png'
    src = cv2.imread(cv2.samples.findFile(frame_path), cv2.IMREAD_COLOR)

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
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, color)
   
    # SAVE TO FILE (OVERWRITING THE BUFFER IS OK)
    cv2.imwrite(frame_path, dst)


# PART II DEFINITIONS

def poc_one_deploy():

    '''TODO DOCU - Function to test the model with a batch of images and show the labels predictions'''

    # stdout
    print("\n***************************************************************************************************************************\n" + 
          " CNN Deployment - PoC #1: Image classification WITHOUT localization (That is, adding bounding boxes to the captured images)" +
          "\n***************************************************************************************************************************\n")

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

    # TODO -> DOCU
    classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

    # TODO -> DOCU
    print( " - The number of images in a deploy set is: " + str(len(deploy_data)) + "\n" )
    print( " - The batch-size is: " + str(BATCH_SIZE) + "\n" )
    print( " - The number of batches needed to procees the whole deployment data is:" + str(int(len(deploy_data)//BATCH_SIZE)) + "\n\n" ) # DEBUG - There is a first grade's misscalculation here ^^' use mod op (%) or whatever suits better

    #######################################################################
    # STEP 2: Loading the CNN model and importing the data  
    #######################################################################

    # Let's load the model that got best accuracy during training (86%) for 
    # the PoC-ONE dataset:

    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - Model deployed on", device, "device")

    # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # DEBUG - THE LOOP STARTS HERE
    # for j in range(len(deploy_data)):
    mydata = []
    frame_id = 0 #COUNTER

    for i, (images, labels) in enumerate(deploy_loader, 0):
    
        # get batch of images from the test DataLoader  
        images, labels = next(iter(deploy_loader)) 

        print("BATCH NUMBER",i)
        
        # MOVING DATA TO THE GPU MEM SPACE
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # show all images as one image grid
        img_grid = Tensor.cpu(torchvision.utils.make_grid(images))
        img_grid = img_grid / 2 + 0.5  # unnormalize
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(POC_FOLDER+'out/labels-prediction.png')
        plt.show()

        #######################################################################
        # STEP 3: Peforming predictions over the imported data  
        #######################################################################
        

        # Let's see what if the model identifies the labels of these example
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        #######################################################################
        # STEP 4: Showing results  
        #######################################################################

        # Even though we can't process 

        # Preparing the data to build a table 

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
            print("(frame_id,i,j,prediction)",frame_id,i,j,predicted_label)
            mydata.append([frame_id, predicted_label])
            frame_id += 1
    
    # create header
    head = ["Frame ID", "Classification Result"]
    
    # print the table to info file
    print(tabulate(mydata, headers=head, tablefmt="grid"), file=PREDICTIONS_OUT_FILE)

    for i in range(len(deploy_data)): # TODO DEBUG - THIS IS A PATCH! We're creating a myData list of size multiple of the batch-size, instead of the deploy data size
        # TODO - DOC
        add_bounding_box(mydata[i])

def poc_one_deploy_v07_14():

    '''TODO DOCU - Function to test the model with a batch of images and show the labels predictions'''

    # stdout
    print("\n***************************************************************************************************************************\n" + 
          " CNN Deployment - PoC #1: Image classification WITHOUT localization (That is, adding bounding boxes to the captured images)" +
          "\n***************************************************************************************************************************\n")

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

    # TODO -> DOCU
    classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

    # TODO -> DOCU
    print( " - The number of images in a deploy set is: " + str(len(deploy_data)) + "\n" )
    print( " - The batch-size is: " + str(BATCH_SIZE) + "\n" )
    print( " - The number of batches needed to procees the whole deployment data is:" + str(int(len(deploy_data)//BATCH_SIZE)) + "\n\n" ) # DEBUG - There is a first grade's misscalculation here ^^' use mod op (%) or whatever suits better

    #######################################################################
    # STEP 2: Loading the CNN model and importing the data  
    #######################################################################

    # Let's load the model that got best accuracy during training (86%) for 
    # the PoC-ONE dataset:
    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - Model deployed on", device, "device")

    # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # DEBUG - THE LOOP STARTS HERE
    # for j in range(len(deploy_data)):
    mydata = []
    k = 0 #COUNTER
    for i, (images, labels) in enumerate(deploy_loader, 0):
    
    # get batch of images from the test DataLoader  
        images, labels = next(iter(deploy_loader)) 
        
        # MOVING DATA TO THE GPU MEM SPACE
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # show all images as one image grid
        img_grid = Tensor.cpu(torchvision.utils.make_grid(images))
        img_grid = img_grid / 2 + 0.5  # unnormalize
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(POC_FOLDER+'out/labels-prediction.png')
        plt.show()

        #######################################################################
        # STEP 3: Peforming predictions over the imported data  
        #######################################################################
        

        # Let's see what if the model identifies the labels of these example
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        #######################################################################
        # STEP 4: Showing results  
        #######################################################################

        # Even though we can't process 

        # Preparing the data to build a table 

        for j in range(len(predicted)):

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
            frame_id = k # DEBUG: It would be nice to add some formatting (zeroes padding at left, etc)
            predicted_label = classes[predicted[j]]
            mydata.append([frame_id, predicted_label])
            k += 1
    
    # create header
    head = ["Frame ID", "Classification Result"]
    
    # print the table to info file
    print(tabulate(mydata, headers=head, tablefmt="grid"), file=PREDICTIONS_OUT_FILE)

    for i in range(len(deploy_data)): # TODO DEBUG - THIS IS A PATCH! We're creating a myData list of size multiple of the batch-size, instead of the deploy data size
        # TODO - DOC
        add_bounding_box(mydata[i])

def main():

    '''TODO DOCU - Just a plain-good'ol main function'''

    ###########################################################################
    ###############################      PART I      ##########################
    ###########################################################################
    ##
    ##      Simulating FLIR camera video capture (0.5 FPS) in real-time  
    ##
    ###########################################################################

    # Let's run the night/thermal-vision simulation on the images taken by the drone on it's LATEST flight
    thermal_vision_simulation()

    ############################################################################
    ###############################      PART II     ###########################
    ############################################################################
    ##
    ##  PoC #1: Image classification WITHOUT adding bounding boxes to the FLIR images, 
    ##          which contain wilfire instances from multiples classes
    ##
    ##  https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
    ##  https://medium.com/analytics-vidhya/guide-to-object-detection-using-pytorch-3925e29737b9 
    ##
    ###########################################################################

    poc_one_deploy()


    # TODO - CREATING FINAL VIDEO
    '''
    pathIn= FLIR_BUFFER + 'unknown-class/'
    pathOut = POC_FOLDER + 'out/video.avi'
    fps = 20.0
    convert_frames_to_video(pathIn, pathOut, fps)
    '''

# CALLING MAIN.
if __name__ == '__main__':
    main()
