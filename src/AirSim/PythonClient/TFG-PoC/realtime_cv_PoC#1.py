"""
Title::

    pytorch_deployment_PoC#1.py 

Python environment: 

    TODO Python 3.6.4 + pytorch + CUDA, etc....

Description::

    Deploying the classification algorithm (model cnn-training_v3.pth) without localization boxes (just image classification)

Inputs::

    TODO

Output::

    TODO


Original author::

    Chris Lovett - https://github.com/lovettchris

Modified / adapted by:   

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#MultirotorClient.moveOnPathAsync 
    2 - https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo 


TODO list: 

"""



# IMPORTS

## PART I 


import cv2 # https://stackoverflow.com/questions/50909569/unable-to-import-cv2-module-python-3-6
import os
import time
import glob
import shutil

from tabulate import tabulate
from torch.functional import Tensor
import torchvision

## PART II 

# DEBUG: We copied the file into this project folder in a rush. Create a shared lib in /usr/lib by instance....

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

# CONSTANTS AND LITERALS

# PART I

UE4_ZONE_6 = 6
UE4_ZONE_7 = 7

# PART II

#data-bond constants
DEPLOY_DATA_DIR = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer"
DEPLOY_IMG_SIZE = 229
# BATCH_SIZE = 256 # DEBUG: No batches today. We're just loading the whole deployment dataset into the GPU memory. It won't be that much for the PoC... will it?

#model-bond constants
MODEL_VERSION = 3
MODEL_PATH = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/"

# FUNCTION DEFINITIONS

# PART I DEFINITIONS

def getNewestItem(path):
    '''
    Function that return the newest item in a base path
    Sources -> https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder 
    '''
    list_of_files = glob.glob(path+'*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def create_flir_img(thermal_img_path, rgb_img_path, composite_img_path, ue4_zone):
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
            Thermal+RGB composite image (file)

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
    
    # Extracting the width and height 
    # of the image (both images are equal in size): --> https://appdividend.com/2020/09/09/python-cv2-image-size-how-to-get-image-size-in-python/
    height, width = grayscale_image.shape

    # We must set a filter to discard the images that did not show any of the virtual 
    # wildfire features captured by the built-in AirSim infrared camera simulator -that is, 
    # images that do not include ANY white pixels. For this purpose, we'll be using the 
    # 'fire_img' bool variable set as False by default, and then we'll set it to true if 
    # there are pixels with their RGB value as 255, which simulate the fire detected by the 
    # FLIR camera simulator 

    fire_img = False
  
    for i in range(height):
        for j in range(width):

            # getting the THERMAL pixel value.
            p = thermal_image[i,j]

            # If the pixel is WHITE (#FFFFFF) then it's hot! -> Therefore we set the #FFFFFF=255 
            # value on the RGB image (scene) HOWEVER, if we're going to take images WITHOUT 
            # wildfires (UE_ZONE == 7), then we don't set the filter up
            # DEBUG: IN THIS IMPLEMENTATION...
            if (p==255) and ue4_zone != UE4_ZONE_7:
                grayscale_image[i, j] = p           
                fire_img = True

    # Saving the final output -- DEBUG -> pending to set a relative path 
    # We discard images with no white pixels, except in the case we are 
    # taking no-wildfire images (zone 7)
    if fire_img or ue4_zone == UE4_ZONE_6 or ue4_zone == UE4_ZONE_7:
        cv2.imwrite(composite_img_path+'/FLIR_'+ time.strftime('%Y%m%d-%H%M%S') +'.png',grayscale_image)

def thermal_vision_simulation():
    
    ### TODO -> DOCUMENTATION

    ########################################################################### 
    # STEP 1: Setting path variables
    ###########################################################################

    # Base folder
    poc_folder = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 

    # INPUT sample folder
    sample_folder = getNewestItem(poc_folder + 'in/') + '/'

    # OUTPUT to FLIR conversion buffer
    flir_buffer = poc_folder + 'flir_buffer/unknown-class/'


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
        # STEP 3: Generating the FLIR image on "almost real-time"
        #######################################################################

        ### TODO -> DOCUMENTATION
        segment_img_path = sample_folder + 'images/' + mySegment_file
        rgb_img_path = sample_folder + 'images/' + myRGB_file

        ### TODO -> DOCUMENTATION
        create_flir_img(segment_img_path, rgb_img_path, flir_buffer, UE4_ZONE_6)
        
        #Deleting input files
        os.remove(segment_img_path)
        os.remove(rgb_img_path)

    #Deleting samples folder
    shutil.rmtree(sample_folder) 

# PART II DEFINITIONS

# Function to test the model with a batch of images and show the labels predictions
def poc_one_deploy():

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
        transforms.RandomResizedCrop(512),
        # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
        transforms.Resize(DEPLOY_IMG_SIZE)
    ])

    # Create an instance for deployment
    deploy_data = (datasets.ImageFolder(root=DEPLOY_DATA_DIR, transform=transformations))

    # Create a loader for the test set which will read the data within batch size and put into memory. 
    # Note that each shuffle is set to false, since we will be creating a frame-by-frame animation 
    deploy_loader = DataLoader(deploy_data, batch_size=len(deploy_data), shuffle=False, num_workers=0)

    classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

    print(" - The number of images in a deploy set is: ", len(deploy_data))

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

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

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
    plt.savefig(MODEL_PATH + '/labels-prediction.png')
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

    # Showing the accuracy of the final test - DEBUG - We don't do accuracy stats on deployment
    # STATS["final_test_acc"].append(calculateAccuracy(deploy_loader, model))
    # print(" - Final test accuracy: %d %%" % (STATS["final_test_acc"][0])  ,"\n")

    # Preparing the data to build a table 
    mydata = []
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
        frame_id = j # DEBUG: It would be nice to add some formatting (zeroes padding at left, etc)
        predicted_label = classes[predicted[j]]
        mydata.append([frame_id, predicted_label])
    
    # create header
    head = ["Frame ID", "Classification Result"]
    
    # print the table to info file
    print(tabulate(mydata, headers=head, tablefmt="grid"))  

if __name__ == '__main__':

    ###########################################################################
    ###############################      PART I      ##########################
    ###########################################################################
    ##
    ##      Simulating FLIR camera video capture (0.5 FPS) in real-time  
    ##
    ###########################################################################

    # Let's run the night/thermal-vision simulation on the images taken by the drone on it's LATEST flight
    #thermal_vision_simulation()

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