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
        # STEP 3: Generating the FLIR image on real
        #######################################################################

        # TODO -> WORK IN PROGRESS...
        segment_img_path = sample_folder + 'images/' + mySegment_file
        rgb_img_path = sample_folder + 'images/' + myRGB_file

        create_flir_img(segment_img_path, rgb_img_path, flir_buffer, UE4_ZONE_6)
        
        #Deleting input files
        os.remove(segment_img_path)
        os.remove(rgb_img_path)

    ############################################################################
    ###############################      PART II     ###########################
    ############################################################################
