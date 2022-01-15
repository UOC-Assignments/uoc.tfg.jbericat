"""
title:: 

    Airsim related auxiliar functions

description::


inputs::


output::


author::

    Jordi Bericat Ruz 
    other sources


references::

"""

###############################################################################

# Import this module to automatically setup path to local airsim module
# This module first tries to see if airsim module is installed via pip
# If it does then we don't do anything else
# Else we look up grand-parent folder to see if it has airsim folder
#    and if it does then we add that in sys.path

import os,sys,logging
#from PIL import Image

#this class simply tries to see if airsim 
class SetupPath:
    @staticmethod
    def getDirLevels(path):
        path_norm = os.path.normpath(path)
        return len(path_norm.split(os.sep))

    @staticmethod
    def getCurrentPath():
        cur_filepath = __file__
        return os.path.dirname(cur_filepath)

    @staticmethod
    def getGrandParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 2:
            return os.path.dirname(os.path.dirname(cur_path))
        return ''

    @staticmethod
    def getParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 1:
            return os.path.dirname(cur_path)
        return ''

    @staticmethod
    def addAirSimModulePath():
        # if airsim module is installed then don't do anything else
        #import pkgutil
        #airsim_loader = pkgutil.find_loader('airsim')
        #if airsim_loader is not None:
        #    return

        parent = SetupPath.getParentDir()
        if parent !=  '':
            airsim_path = os.path.join(parent, 'airsim')
            client_path = os.path.join(airsim_path, 'client.py')
            if os.path.exists(client_path):
                sys.path.insert(0, parent)
        else:
            logging.warning("airsim module not found in parent folder. Using installed package (pip install airsim).")

SetupPath.addAirSimModulePath()

###############################################################################


import cv2 as cv

UE4_ZONE_0 = 0
UE4_ZONE_1 = 1
UE4_ZONE_2 = 2
UE4_ZONE_3 = 3
UE4_ZONE_4 = 4
UE4_ZONE_5 = 5
UE4_ZONE_6 = 6
UE4_ZONE_7 = 7
UE4_ZONE_8 = 8



import os
import glob

def get_newest_item(path):
    '''
    Function that return the newest item in a base path
    Sources -> https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder 
    '''
    list_of_files = glob.glob(path+'*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

###############################################################################

import shutil

# Base folders
POC_FOLDER = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 
FLIR_BUFFER = POC_FOLDER + 'flir_buffer/'
PREDICTIONS_OUT_FILE = open(os.path.abspath(POC_FOLDER + 'out/frame-predictions.log'), "w")   
DEPLOY_DATA_DIR = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer"

def flir_offline_batch_coverter():
    
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
        myRGB_file = log_entry_filenames[1] # DEBUG _ THIS IS 1 NOW!
        mySegment_file  = log_entry_filenames[0] # DEBUG _ THIS IS 0 NOW!

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

###############################################################################


# Refs ->  https://airsim-fork.readthedocs.io/en/latest/image_apis.html#segmentation

import numpy
import time
import sys
import os
from airsim import *

def rotation_matrix_from_angles(pry):
    pitch = pry[0]
    roll = pry[1]
    yaw = pry[2]
    sy = numpy.sin(yaw)
    cy = numpy.cos(yaw)
    sp = numpy.sin(pitch)
    cp = numpy.cos(pitch)
    sr = numpy.sin(roll)
    cr = numpy.cos(roll)
    
    Rx = numpy.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    
    Ry = numpy.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    
    Rz = numpy.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    
    #Roll is applied first, then pitch, then yaw.
    RyRx = numpy.matmul(Ry, Rx)
    return numpy.matmul(Rz, RyRx)

def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    #Turn the camera position into a column vector.
    camPosition = numpy.transpose([camXYZ])

    #Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = utils.to_eularian_angles(camQuaternion)
    
    #Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = rotation_matrix_from_angles(pitchRollYaw)
    
    #Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = numpy.transpose([subjectXYZ])
    XYZW = numpy.add(XYZW, -camPosition)
    print("XYZW: " + str(XYZW))
    XYZW = numpy.matmul(numpy.transpose(camRotation), XYZW)
    print("XYZW derot: " + str(XYZW))
    
    #Recreate the perspective projection of the camera.
    XYZW = numpy.concatenate([XYZW, [[1]]])    
    XYZW = numpy.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]
    
    #Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2
    
    return numpy.array([
        imageWidthHeight[0] * normX,
        imageWidthHeight[1] * normY
    ]).reshape(2,)
   
def get_image(camera, x, y, z, pitch, roll, yaw, client):
    """
    title::
        get_image

    description::
        Capture images (as numpy arrays) from a certain position.

    inputs::
        x
            x position in meters
        y
            y position in meters
        z
            altitude in meters; remember NED, so should be negative to be 
            above ground
        pitch
            angle (in radians); in computer vision mode, this is camera angle
        roll
            angle (in radians)
        yaw
            angle (in radians)
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)

    returns::
        position
            AirSim position vector (access values with x_val, y_val, z_val)
        angle
            AirSim quaternion ("angles")
        im
            segmentation or IR image, depending upon palette in use (3 bands)
        imScene
            scene image (3 bands)

    author::
        Elizabeth Bondi
        Shital Shah
    """

    # Set pose and sleep after to ensure the pose sticks before capturing 
    # image AND to wait for the fire VFX objects to start rendering.
    client.simSetVehiclePose(Pose(Vector3r(x, y, z), \
                      to_quaternion(pitch, roll, yaw)), True)
    time.sleep(0.1)

    # Capture segmentation (IR) and scene images.
    responses = \
        client.simGetImages([ImageRequest(camera, ImageType.Infrared,
                                          False, False),
                            ImageRequest(camera, ImageType.Scene, \
                                          False, False),
                            ImageRequest(camera, ImageType.Segmentation, \
                                          False, False)])

    #Change images into numpy arrays.

   ## S'HA HAGUT DE MODIFICAR EL CODI PER A QUE COMPILI!! (el darrer paràmetre 
   # de .reshape ha de ser 3 i no 4, es podria fer un pull request al repo de AirSim) 
    img1d = numpy.fromstring(responses[0].image_data_uint8, dtype=numpy.uint8)
    im = img1d.reshape(responses[0].height, responses[0].width, 3) 

    img1dscene = numpy.fromstring(responses[1].image_data_uint8, dtype=numpy.uint8)
    imScene = img1dscene.reshape(responses[1].height, responses[1].width, 3)

    return Vector3r(x, y, z), to_quaternion(pitch, roll, yaw),\
           im[:,:,:3], imScene[:,:,:3] #get rid of alpha channel

def radiance(absoluteTemperature, emissivity, dx=0.01, response=None):
    """
    title::
        radiance

    description::
        Calculates radiance and integrated radiance over a bandpass of 8 to 14
        microns, given temperature and emissivity, using Planck's Law.

    inputs::
        absoluteTemperature
            temperture of object in [K]

            either a single temperature or a numpy
            array of temperatures, of shape (temperatures.shape[0], 1)
        emissivity
            average emissivity (number between 0 and 1 representing the
            efficiency with which it emits radiation; if 1, it is an ideal 
            blackbody) of object over the bandpass

            either a single emissivity or a numpy array of emissivities, of 
            shape (emissivities.shape[0], 1)
        dx
            discrete spacing between the wavelengths for evaluation of
            radiance and integration [default is 0.1]
        response
            optional response of the camera over the bandpass of 8 to 14 
            microns [default is None, for no response provided]
    
    returns::
        radiance
            discrete spectrum of radiance over bandpass
        integratedRadiance
            integration of radiance spectrum over bandpass (to simulate
            the readout from a sensor)

    author::
        Elizabeth Bondi
    """
    wavelength = numpy.arange(8,14,dx)
    c1 = 1.19104e8 # (2 * 6.62607*10^-34 [Js] * 
                   # (2.99792458 * 10^14 [micron/s])^2 * 10^12 to convert 
                   # denominator from microns^3 to microns * m^2)
    c2 = 1.43879e4 # (hc/k) [micron * K]
    if response is not None:
        radiance = response * emissivity * (c1 / ((wavelength**5) * \
                   (numpy.exp(c2 / (wavelength * absoluteTemperature )) - 1)))
    else:
        radiance = emissivity * (c1 / ((wavelength**5) * (numpy.exp(c2 / \
                   (wavelength * absoluteTemperature )) - 1)))
    if absoluteTemperature.ndim > 1:
        return radiance, numpy.trapz(radiance, dx=dx, axis=1)
    else:
        return radiance, numpy.trapz(radiance, dx=dx)

def get_new_temp_emiss_from_radiance(tempEmissivity, response):
    """
    title::
        get_new_temp_emiss_from_radiance

    description::
        Transform tempEmissivity from [objectName, temperature, emissivity]
        to [objectName, "radiance"] using radiance calculation above.

    input::
        tempEmissivity
            numpy array containing the temperature and emissivity of each
            object (e.g., each row has: [objectName, temperature, emissivity])
        response
            camera response (same input as radiance, set to None if lacking
            this information)

    returns::
        tempEmissivityNew
            tempEmissivity, now with [objectName, "radiance"]; note that 
            integrated radiance (L) is divided by the maximum and multiplied 
            by 255 in order to simulate an 8 bit digital count observed by the 
            thermal sensor, since radiance and digital count are linearly 
            related, so it's [objectName, simulated thermal digital count]

    author::
        Elizabeth Bondi
    """
    numObjects = tempEmissivity.shape[0]

    L = radiance(tempEmissivity[:,1].reshape((-1,1)).astype(numpy.float64), 
                 tempEmissivity[:,2].reshape((-1,1)).astype(numpy.float64), 
                 response=response)[1].flatten() 
    L = ((L / L.max()) * 255).astype(numpy.uint8)

    tempEmissivityNew = numpy.hstack((
        tempEmissivity[:,0].reshape((numObjects,1)), 
        L.reshape((numObjects,1))))

    return tempEmissivityNew

def set_segmentation_ids(segIdDict, tempEmissivityNew, client):
    """
    title::
        set_segmentation_ids

    description::
        Set stencil IDs in environment so that stencil IDs correspond to
        simulated thermal digital counts (e.g., if elephant has a simulated
        digital count of 219, set stencil ID to 219).

    input::
        segIdDict
            dictionary mapping environment object names to the object names in
            the first column of tempEmissivityNew 
        tempEmissivityNew
            numpy array containing object names and corresponding simulated
            thermal digital count
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)

    author::
        Elizabeth Bondi
    """

    #First set everything to 0.
    success = client.simSetSegmentationObjectID("[\w]*", 0, True);
    if not success:
        print('There was a problem setting all segmentation object IDs to 0. ')
        sys.exit(1)

    #Next set all objects of interest provided to corresponding object IDs
    #segIdDict values MUST match tempEmissivityNew labels.
    for key in segIdDict:
        objectID = int(tempEmissivityNew[numpy.where(tempEmissivityNew == \
                                                     segIdDict[key])[0],1][0])

        success = client.simSetSegmentationObjectID("[\w]*"+key+"[\w]*", 
                                                    objectID, True);
        if not success:
            print('There was a problem setting {0} segmentation object ID to {1!s}, or no {0} was found.'.format(key, objectID))
            
    time.sleep(0.1)

def set_environment(client):
    # TFG: Comment #03.xx
    # 
    # On the UE4 environment created to perform this project's 
    # proof of concept -based on the LandscapeMountains env-, 
    # the "StaticMeshActor" objects that include the strings 
    # "fire" and "grass" on its object name are the ones that
    # simulate heating emission. 

    segIdDict = {'grass':'big_fire', #DEBUG: Assignem una temperatura (color en la escala de grisos) als pixels corresponents a VFX foc
                'firewood':'small_fire'} #DEBUG: Assignem una temperatura (color en la escala de grisos) als pixels corresponents a vegetació (arbres)
    
    #Choose temperature values 
    tempEmissivity = numpy.array([['big_fire',298,0.98], 
                                  ['small_fire',298,0.98]])

    #Read camera response.
    response = None
    camResponseFile = 'camera_response.npy'
    try:
      numpy.load(camResponseFile)
    except:
      print("{} not found. Using default response.".format(camResponseFile))

    #Calculate radiance.
    tempEmissivityNew = get_new_temp_emiss_from_radiance(tempEmissivity, 
                                                         response)

    #Set IDs in AirSim environment.
    set_segmentation_ids(segIdDict, tempEmissivityNew, client)

# Image class labeling - We use the dataset's directory tree structure to set 
# the image class labeling depending on the UE4 environment zone we're 
# taking the pictures of. We can implement this using an elif 
# ladder, since there is no buit-in switch construct in Python 
# -> https://pythongeeks.org/switch-in-python/


###############################################################################
#CUSTOM FUNCTIONS

###############################################################################

def set_class_folder(input):
    if (input == UE4_ZONE_0):
        no_wildfires = False
        selection = 'test/high-intensity-wildfires/'
        
    elif (input == UE4_ZONE_1):
        no_wildfires = False
        selection = 'training+validation/high-intensity-wildfires/'

    elif (input == UE4_ZONE_2): 
        no_wildfires = False
        selection = 'test/medium-intensity-wildfires/'

    elif (input == UE4_ZONE_3):
        no_wildfires = False
        selection = 'training+validation/medium-intensity-wildfires/'

    elif (input == UE4_ZONE_4): 
        no_wildfires = False
        selection = 'test/low-intensity-wildfires/'

    elif (input == UE4_ZONE_5):
        no_wildfires = False
        selection = 'training+validation/low-intensity-wildfires/'

    elif (input == UE4_ZONE_6):
        no_wildfires = True
        selection = 'test/no-wildfires/'

    elif (input == UE4_ZONE_7):
        no_wildfires = True
        selection = 'training+validation/no-wildfires/'

    return selection, no_wildfires

def create_flir_img_v1(thermal_img_path, rgb_img_path, composite_img_path, ue4_zone):
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
        https://stackoverflow.com/questions/48190894/how-to-convert-rgb-images-dataset-to-single-channel-grayscale
    """
    # Import an image from directory:
    thermal_image = cv.imread(thermal_img_path, cv.IMREAD_GRAYSCALE)
    rgb_image = cv.imread(rgb_img_path, cv.IMREAD_COLOR)
    #rgb_image = Image.open(rgb_img_path)

    # Convert RGB image to grayscale -> https://stackoverflow.com/questions/48190894/how-to-convert-rgb-images-dataset-to-single-channel-grayscale
     
    # When using opencv, we load images into the BGR color space. Therefore, we convert BGR -> GRAY instead of RGB -> GRAY
    grayscale_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY) 
    
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
    # We discard images with no white pixels, except in the case we are taking no-wildfire images (zone 7)
    if fire_img or ue4_zone == UE4_ZONE_6 or ue4_zone == UE4_ZONE_7:
        cv.imwrite(composite_img_path,grayscale_image)

import numpy as np

def create_flir_img_v2(thermal_img_path, rgb_img_path, composite_img_path, deploy):
    '''
    title::
        create_flir_img VERSION 2 (a lot faster than VERSION 1)

    description::
        Thermal + RGB composite image: 'injects' the heat pixels (whites) detected by 
        IR-Thermal into the scene image to simulate a FLIR based thermal vision camera device.

        On this version we're going to try a more efficient approach using the numpy framework for python.
    
        THIS VERSION CANNOT BE USED TO RETRIEVE NO-WILDFIRE CLASS IMAGES (THOSE ARE GOING TO 
        BE RETRIEVED IN MULTICOPTER MODE LATER)

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
        https://stackoverflow.com/questions/48190894/how-to-convert-rgb-images-dataset-to-single-channel-grayscale
        https://www.geeksforgeeks.org/searching-in-a-numpy-array/  

    TODO::
        - Adapt for airsim_capture_environment.py
    '''
    # Import an image from directory:
    thermal_image = cv.imread(thermal_img_path, cv.IMREAD_GRAYSCALE)
    rgb_image = cv.imread(rgb_img_path, cv.IMREAD_COLOR)
    
    # Convert RGB image to grayscale. When using opencv, we load images
    # into the BGR color space. Therefore, we convert BGR -> GRAY 
    # instead of RGB -> GRAY
    grayscale_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY) 

    #  looking for value 255 (white pixel) on the vectorized matrix and storing its index in i
    # TODO - DOC double loop optimization 
    myTuple = np.where(thermal_image == 255)
    if ( len(myTuple[0]) or deploy==True ):
        
        for index in range(len(myTuple[0])):
            x = myTuple[0][index]
            y = myTuple[1][index]
            grayscale_image[x, y] = 255

        # We have to explicitly tell that we want to use the global namespace to 
        # keep track of the frame number we want to append to the output file
        global framecount
        cv.imwrite(composite_img_path, grayscale_image) # DEBUG -> S'HA DE PASSAR EL NOM D'ARXIU SENCER COM A PARÀMETRE
        #cv.imwrite(composite_img_path+'/FLIR_frame-' + str(framecount) + '.png', grayscale_image)

###############################################################################

