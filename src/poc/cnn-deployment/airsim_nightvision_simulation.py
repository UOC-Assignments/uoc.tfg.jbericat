"""
Title::

    pytorch_deployment_PoC#1_v1.0-.py 

Python environment: 

    TODO Python 3.6.4 + pytorch + CUDA, etc....

Description::

Inputs::

Output::

Original author::

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - 

TODO list: 
    1 - 
"""

# IMPORTS

## PART I 

from genericpath import isfile
from posixpath import join
import os
import glob
import shutil
from numpy.core.fromnumeric import take

#from tabulate import tabulate
#from torch.functional import Tensor, meshgrid
#from torch.utils.data.sampler import BatchSampler
#import torchvision

## PART II 

# DEBUG: We copied some code (defs) into this project folder in a rush. Create a shared lib in /usr/lib by instance....

# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder

from poc.lib.create_flir_image import *

#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
import sys

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


'''
def create_flir_img_v2(thermal_img_path, rgb_img_path, composite_img_path, ue4_zone):

    # Import an image from directory:
    thermal_image = cv.imread(thermal_img_path, cv.IMREAD_GRAYSCALE)
    rgb_image = cv.imread(rgb_img_path, cv.IMREAD_COLOR)

    # Convert RGB image to grayscale -> https://stackoverflow.com/questions/48190894/how-to-convert-rgb-images-dataset-to-single-channel-grayscale
     
    # When using opencv, we load images into the BGR color space. Therefore, we convert BGR -> GRAY instead of RGB -> GRAY
    grayscale_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY) 
    
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
    cv.imwrite(composite_img_path+'/FLIR_frame-' + str(framecount) + '.png', grayscale_image)
    framecount += 1 # NO CONCURRENCY = NO NEED TO MUTEX OR LOCKS, BUT THAT COULD CHANGE IN FUTURE VERSIONS
    '''

def thermal_vision_offline():
    
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
    framecount = 0

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
        create_flir_img_v2(segment_img_path, rgb_img_path, str(output) + '/frame-' + str(framecount) + '.png', True)
        framecount += 1 # NO CONCURRENCY = NO NEED TO MUTEX OR LOCKS, BUT THAT COULD CHANGE IN FUTURE VERSIONS
        
        #Deleting input files
        os.remove(segment_img_path)
        os.remove(rgb_img_path)

    #Deleting samples folder
    shutil.rmtree(sample_folder) 

# CALLING MAIN.
if __name__ == '__main__':

    ###########################################################################
    ###############################      PART I      ##########################
    ###########################################################################
    ##
    ##      Simulating FLIR camera video capture (0.5 FPS) in real-time  
    ##
    ###########################################################################

    # Let's run the night/thermal-vision simulation on the images taken by the drone on it's LATEST flight
    thermal_vision_offline()