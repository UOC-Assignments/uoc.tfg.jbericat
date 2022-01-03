# IMPORTS

## PART I 


import cv2 # https://stackoverflow.com/questions/50909569/unable-to-import-cv2-module-python-3-6
import os
import time
import glob
import shutil

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
#BATCH_SIZE = 256 # DEBUG: Why not loading the whole deployment dataset into the GPU memory? It won't be that much...
CONF_THRES = 0.8
NMS_THRES = 0.4

#model-bond constants
MODEL_VERSION = 3
MODEL_PATH = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/"

''''''
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

# Functions that return the newest and oldest created item in a base path
# Sources -> https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder 
def getNewestItem(path):
    list_of_files = glob.glob(path+'*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

if __name__ == '__main__':

    ###########################################################################
    ###############################      PART I      ##########################
    ###########################################################################
    ##
    ##      Simulating FLIR camera video capture (0.5 FPS) in real-time  
    ##
    ###########################################################################

    def thermal_vision_simulation():
    
        ########################################################################### 
        # STEP 1: Setting path variables
        ###########################################################################

        # Base folder
        poc_folder = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 

        # Input sample folder
        sample_folder = getNewestItem(poc_folder + 'in/') + '/'

        # FLIR conversion buffer
        flir_buffer = poc_folder + 'flir_buffer/'

        # out_folder 
        # TODO

        ###########################################################################
        # STEP 2: Extract SEGMENT & RGB Filenames from the airsim_rec.txt log
        ###########################################################################
        
        # Deleting the first line of the file (Header)
        with open(sample_folder + 'airsim_rec.txt', 'r') as fin:
            data = fin.read().splitlines(True)
        with open(sample_folder + 'airsim_rec.txt', 'w') as fout:
            fout.writelines(data[1:])

        ### TODO THE LOOP STARTS HERE!
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

            # TODO -> WORK IN PROGRESS...
            segment_img_path = sample_folder + 'images/' + mySegment_file
            rgb_img_path = sample_folder + 'images/' + myRGB_file

            create_flir_img(segment_img_path, rgb_img_path, flir_buffer, UE4_ZONE_6)
            
            #Deleting input files
            os.remove(segment_img_path)
            os.remove(rgb_img_path)

        #Deleting samples folder
        shutil.rmtree(sample_folder) 

    # Let's run the night/thermal-vision simulation on the images taken by the drone on it's LATEST flight
    #thermal_vision_simulation()

    ############################################################################
    ###############################      PART II     ###########################
    ############################################################################
    ##
    ##  PoC #3: Object detection and tracking_ Adding bounding boxes to the FLIR images, 
    ##          which contain wilfire instances from multiples classes
    ##
    ##  https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
    ##  https://medium.com/analytics-vidhya/guide-to-object-detection-using-pytorch-3925e29737b9 
    ##
    ###########################################################################

    #######################################################################
    # STEP 1: Loading the CNN model  
    #######################################################################

    # Let's load the model that got best accuracy (86%) for the PoC dataset:
    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - Model tested on", device, "device")

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # Set the model on evaluation (so it won't be reajusting any weights)
    model.eval

    # TODO - DOCUMENTATION - Still need to figure out how to define exactly a TENSOR....
    Tensor = torch.cuda.FloatTensor

    #######################################################################
    # STEP 2: Defining the detection function
    #######################################################################

    # Define the classes contained on the deployment dataset
    classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

    def detect_image(img):
        # scale and pad image
        
        ratio = min(DEPLOY_IMG_SIZE/img.size[0], DEPLOY_IMG_SIZE/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms=transforms.Compose([
            
            #transforms.Resize((imh,imw)),
            #transforms.Pad((max(int((imh-imw)/2),0), 
            #    max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
            #    max(int((imw-imh)/2),0)), (128,128,128)),
                
            transforms.ToTensor(),
            ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = utils.non_max_suppression(detections, 80, 
                            CONF_THRES, NMS_THRES)
        return detections[0]

    #######################################################################
    # STEP 3: Putting-it-all together
    #######################################################################

    # load image and get detections
    img_path = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer/FLIR_20211231-212332.png" # DEBUG: TEST IMAGE
    prev_time = time.time()
    img = Image.open(img_path)
    detections = detect_image(img)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print ('Inference Time: %s' % (inference_time))

    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)
    
    pad_x = max(img.shape[0] - img.shape[1], 0) * (DEPLOY_IMG_SIZE / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (DEPLOY_IMG_SIZE / max(img.shape))
    unpad_h = DEPLOY_IMG_SIZE - pad_y
    unpad_w = DEPLOY_IMG_SIZE - pad_x

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        # browse detections and draw bounding boxes
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            color = bbox_colors[int(np.where(
                unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=classes[int(cls_pred)], 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
    plt.axis('off')
    # save image
    plt.savefig(img_path.replace(".jpg", "-det.jpg"),        
                    bbox_inches='tight', pad_inches=0.0)
    plt.show()

