"""
Title::

    .py 

Conda environment: "condapy373"

    Python 3.7.3 
    Airsim 1.6

Description::


Inputs::


Output::


Original author::


Modified / adapted by:   


References::

    1 - 

TODO list: 

    1 - DOC

"""

# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 

from poc.lib.airsim import *

import numpy
import cv2
import time

from airsim import *

# METHODS

def batch_capture(client,
        objectList,
        ue4_zone,
        camera,
        height,
        pitch,
        roll=0,
        yaw=0,
        writeIR=True,
        writeScene=True,
        datasetFolder='',
        irFolder='SEGMENT/',
        sceneFolder='RGB/',
        compositeFolder='FLIR/'):
    """
    title::
        main

    description::
        Follow objects of interest and record images while following.

    inputs::
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)
        objectList
            list of tag names within the AirSim environment, corresponding to 
            objects to follow (add tags by clicking on object, going to 
            Details, Actor, and Tags, then add component)
        pitch
            angle (in radians); in computer vision mode, this is camera angle
        roll
            angle (in radians)
        yaw
            angle (in radians)
        z
            altitude in meters; remember NED, so should be negative to be 
            above ground
        write
            if True, will write out the images
        folder
            path to a particular folder that should be used (then within that
            folder, expected folders are ir and scene)

    original author::
        Elizabeth Bondi
    modified by::
        Jordi Bericat Ruz - Universitat Oberta de Catalunya

    """
    i = 0
    for o in objectList:
        pose = client.simGetObjectPose(o);
        
        #Capture image - pose.position x_val access may change w/ AirSim
        #version (pose.position.x_val new, pose.position[b'x_val'] old)

        #TODO - If we're taking no-wildfires images, we should randomly shift-off the camera's x,y & z pos in order to obtain the "richest" sample collection possible

        vector, angle, ir, scene = get_image(camera,
                                pose.position.x_val, 
                                pose.position.y_val, 
                                height, 
                                numpy.radians(pitch), 
                                roll, 
                                yaw, 
                                client)

        # Setting the class label (structured in a folder tree)
        class_folder, no_wildfires = set_class_folder(ue4_zone)
        
        # Adding positional metadata into the images filename and saving into the class folder
        thermal_img_path = (datasetFolder + 
                                class_folder + 
                                irFolder +                                                
                                'SEGMENT_'+ 
                                str(i).zfill(5) + '_' +
                                str('%.5f'%(pose.position.x_val)) + '_' +
                                str('%.5f'%(pose.position.y_val)) + '_' +
                                str(camera) + '_' +
                                str(height) + '_' +
                                str(pitch) +
                                '.png')
        
        rgb_img_path = (datasetFolder + 
                                class_folder + 
                                sceneFolder + 
                                'RGB_' +
                                str(i).zfill(5) + '_' +
                                str('%.5f'%(pose.position.x_val)) + '_' +
                                str('%.5f'%(pose.position.y_val)) + '_' +
                                str(camera) + '_' +
                                str(height) + '_' +
                                str(pitch) +
                                '.png')

        composite_img_path = (datasetFolder + 
                                class_folder + 
                                compositeFolder +
                                'FLIR_' +
                                str(i).zfill(5) + '_' +
                                str('%.5f'%(pose.position.x_val)) + '_' +
                                str('%.5f'%(pose.position.y_val)) + '_' +
                                str(camera) + '_' +
                                str(height) + '_' +
                                str(pitch) +
                                '.png')
        
        if writeIR:
            cv2.imwrite(thermal_img_path, ir)
        if writeScene:
            cv2.imwrite(rgb_img_path, scene)

        if no_wildfires == True:
            # If we're taking no-wildfire images, there is no need to run the 
            # create_flir_img function. Instead, we just have to convert the RGB image
            # to grayscale, which is cheaper
            grayscale = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)  
            cv2.imwrite(composite_img_path, grayscale)

        else:
            # with the "create_flir_img" method we combine both the RGB and SEGMENT 
            # images, obtaining the simulated FLIR thermal vision images as a result.
            create_flir_img_v2(thermal_img_path,rgb_img_path,composite_img_path,False)

        i += 1
        pose = client.simGetObjectPose(o);
        camInfo = client.simGetCameraInfo("0")
        object_xy_in_pic = project_3d_point_to_screen(
            [pose.position.x_val, pose.position.y_val, pose.position.z_val],
            [camInfo.pose.position.x_val, camInfo.pose.position.y_val, camInfo.pose.position.z_val],
            camInfo.pose.orientation,
            camInfo.proj_mat.matrix,
            ir.shape[:2][::-1]
        )
        print("Object projected to pixel\n{!s}.".format(object_xy_in_pic))

if __name__ == '__main__':
    
    #Connect to AirSim, UAV mode.
    client = MultirotorClient()
    client.confirmConnection()

    # SET custom StaticMeshObjects thermal emitters
    set_environment(client)

    # Retrieve custom parameters from std input: Drone camera, height, pitch, roll, yall & UE4 environment zone
    wrong_option=True;
    while (wrong_option):
        print("Specify the class of the simulated night/thermal vision wildfire images you want to generate:\n\n",
                "Zone 0 (Class 1: high intensity wildfire images - small size area) = 0\n",
                "Zone 1 (Class 1: high intensity wildfire images - big size area) = 1\n",
                "Zone 2 (Class 2: medium intensity wildfire images - small size area) = 2\n",
                "Zone 3 (Class 2: medium intensity wildfire images - big size area) = 3\n",
                "Zone 4 (Class 3: low intensity wildfire images - small size area) = 4\n",
                "Zone 5 (Class 3: low intensity wildfire images - big size area) = 5\n",
                "Zone 6 (Class 4: images with no wildfires - small size area) = 6\n",
                "Zone 7 (Class 4: images with no wildfires - big size area) = 7\n",
                "Zone 8 (Class 1+2+3: PoC experiments zone = 8\n")
        ue4_zone = int(input("Please choose an option (0-8 - Default = 1): ") or '1')

        # We control data input correctness and stuff...

        if ue4_zone==UE4_ZONE_6 or ue4_zone==UE4_ZONE_7:
            print("\nIMPORTANT NOTICE: To retrieve no-wildfire images you MUST load the ***UE-no-wildfires-map_dataset-generator*** UE4 file.\n")
            #time.sleep(4)
        
        if ue4_zone==UE4_ZONE_8:
            print("\nERROR: Zone reserved to perfom the PoC experiments (so we avoid overfitting the model by memorizing features).\n")
            time.sleep(4)
        elif ue4_zone<0 or ue4_zone>8:
            print('\nERROR: Wrong option, try again.')
            time.sleep(2)
        else:
            wrong_option=False;

        '''
        wrong_option=True;
        while (wrong_option):
            print("\n\nChoose the multicopter's camera you want to use to retrieve the images:\n\n", 
                    "front_center=0\n",
                    "front_right=1\n",
                    "front_left=2\n",
                    "fpv=3\n",
                    "back_center=4\n")
            camera = int(input("Please choose an option (0-4 - Default = 0): ") or '0')
            if camera<0 or camera>4:
                print('\nERROR: Wrong option, try again.')
                time.sleep(2)
            else:
                wrong_option=False;
        '''

    #pitch = int(input("\n\nSet the camera's pitch angle (Integer degrees 180 > angle > 360 - Default = 270): ") or '270')

    #height = int(input("\n\nSet the multicopter's height (negative integer value - Default = 200 -> lowest height): ") or '-100')

    print("[INFO - ]using the height, camera and pitch lists to get all the images as a batch\n")

    # Look for objects with names that match a regular expression. 
    # On the case of this PoC, we're looking for objects that include 
    # the "firewood" and the "grass" strings on the UE4 env. objects 
    # that simulate heat emission (see the project's report, section 4.6.2).
    #
    # V4.6 -> Enabling the possiblity of generating only specific image classes 
    #         (see section 4.x.x of the project's report) by "injecting" the zone
    #         variable into the regex expression that filters the objects we want
    #         to take pictures of.

    # No-Wildfire / Class #4 images: Since there are no objects tagged as zone == 7 on the Unreal
    # environment, we'll have to use other zones to take class #4 images. 

    if ue4_zone == UE4_ZONE_6:
        my_regex1 = r".*?Rock.*?"
        my_regex2 = r".*?Ice.*?"

    elif ue4_zone == UE4_ZONE_7:
        my_regex1 = r".*?Tree.*?"
        my_regex2 = r".*?Cliffs.*?"
    else:
        my_regex1 = r".*?mesh_firewood_" + str(ue4_zone) + r".*?"
        my_regex2 = r".*?grass_mesh_" + str(ue4_zone) + r".*?"
    
    objectList = client.simListSceneObjects(my_regex1)
    objectList += client.simListSceneObjects(my_regex2)

# CALL TO MAIN

if __name__ == "__main__":

    #Call to main
    heights_list=[-100,-75,-50,-25,0,25,50,75,100]
    cameras_list=[4]
    pitch_list=[255]
    
    for height in heights_list:
        print("************* height", height)
        time.sleep(1)
        
        for camera in cameras_list:
            print("************* camera", camera)

            for pitch in pitch_list:
                time.sleep(1)
                batch_capture(client, 
                    objectList,
                    ue4_zone, 
                    camera,
                    height,
                    pitch, 
                    datasetFolder = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/raw_datasets/buffer/')