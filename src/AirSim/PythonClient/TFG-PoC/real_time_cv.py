"""
title::
    real_time_cv.py

description::


inputs::


output::

author::
    Jordi Bericat Ruz - Universitat Oberta de Catalunya

references::

"""

import cv2
import os
import time
import glob
import shutil

# CONSTANTS AND LITERALS

UE4_ZONE_0 = 0
UE4_ZONE_1 = 1
UE4_ZONE_2 = 2
UE4_ZONE_3 = 3
UE4_ZONE_4 = 4
UE4_ZONE_5 = 5
UE4_ZONE_6 = 6
UE4_ZONE_7 = 7
UE4_ZONE_8 = 8

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

def create_flir_img(thermal_img_path, rgb_img_path, composite_img_path, ue4_zone):
    """
    title::
        create_flir_img

    description::
        Thermal + RGB composite image: "injects" the heat pixels (whites) detected by 
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
    """
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
    # "fire_img" bool variable set as False by default, and then we'll set it to true if 
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
            if (p==255) and ue4_zone != UE4_ZONE_6 and ue4_zone != UE4_ZONE_7:
                grayscale_image[i, j] = p           
                fire_img = True

    # Saving the final output -- DEBUG -> pending to set a relative path 
    # We discard images with no white pixels, except in the case we are 
    # taking no-wildfire images (zone 7)
    if fire_img or ue4_zone == UE4_ZONE_6 or ue4_zone == UE4_ZONE_7:
        cv2.imwrite(composite_img_path+"/FLIR_"+ TIMESTAMP +".png",grayscale_image)

# Functions that return the newest and oldest created item in a base path
# Sources -> https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder 
def getNewestItem(path):
    list_of_files = glob.glob(path+"*") 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def getOldestItem(path):
    list_of_files = glob.glob(path+"*") 
    latest_file = min(list_of_files, key=os.path.getctime)
    return latest_file

if __name__ == "__main__":

    # TODO WHILE THERE ARE FILES ON THE current_drones_captures folder..... DO:

    # First we need to know the folder name where the images are being recorded 
    # (which is the latest created directory by AirSim on the "drone_captures_folder" 
    # when the image recording starts)
    PoC_folder = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/"
    drone_captures_folder = PoC_folder+"drone-realtime-captures/"
    current_drones_captures = getNewestItem(drone_captures_folder)

    # Creating the buffer folder
    if os.path.isdir(current_drones_captures+'/buffer/') == False:
        os.mkdir(current_drones_captures+'/buffer/')

    # Now we need to pair-up the RGB & Segmented IR images taken by the drone. We'll be 
    # using the file creation / modification timestamp to retrieve their filename 
    # (I couldn't figure-out a better way to do so) 
    for i in range(2):
        myItem = getOldestItem(current_drones_captures+"/images/")
        
        if ( int(str(myItem).find("img_Drone1_0_7")) != -1 ):
            # 
            current_SEGMENT_image = getOldestItem(current_drones_captures+"/images/")
            print("latest_IR_image = " + current_SEGMENT_image) # DEBUG

            # moving the oldest SEGMENT image to the processing buffer folder (so, next time we 
            # scan the images folder we won't be retrieving the very same pair of images)
            shutil.move(current_SEGMENT_image, current_drones_captures+'/buffer/tmp_SEGMENT.png')
            segment_img_path = current_drones_captures+'/buffer/tmp_SEGMENT.png'
        
        if ( int(str(myItem).find("img_Drone1_0_0")) != -1 ):

            # if the oldest image file on the current_drones_captures folder is
            # the RGB version of the same image (AirSim type 0), then we retrieve it the same way we did before
            current_RGB_image = getOldestItem(current_drones_captures+"/images/")
            print("latest_RGB_image = " + current_RGB_image) # DEBUG

            # moving the oldest SEGMENT image to the processing buffer folder (so, next time we 
            # scan the images folder we won't be retrieving the very same pair of images)
            shutil.move(current_RGB_image, current_drones_captures+'/buffer/tmp_RGB.png')
            rgb_img_path = current_drones_captures+'/buffer/tmp_RGB.png' 

    # Now is time to create the FLIR simulated image from the ones we send to the buffer folder
    flir_img_path = PoC_folder + "cv-realtime-buffer/"
    create_flir_img(segment_img_path, rgb_img_path, flir_img_path, UE4_ZONE_6)

