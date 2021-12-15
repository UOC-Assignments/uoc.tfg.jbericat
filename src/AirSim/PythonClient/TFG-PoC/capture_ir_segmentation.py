import numpy
import cv2
import time
from PIL import Image
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

    
def create_flir_img(thermal_img_path, rgb_img_path, composite_img_path):
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
    thermal_image = Image.open(thermal_img_path)
    rgb_image = Image.open(rgb_img_path)
    
    # Extracting pixel map:
    rgb_pixel_map = rgb_image.load()
  
    # Extracting the width and height 
    # of the image (both images are equal in size):
    width, height = rgb_image.size

    # We should set a filter to discard the images that did not show any of the virtual 
    # wildfire features captured by the built-in AirSim infrared camera simulator -that is, 
    # images that do not include ANY white pixels. For this purpose, we'll be using the 
    # "fire_img" bool variable set as False by default, and then we'll set it to true if 
    # there are pixels with their RGB value as 255, which simulate the fire detected by the 
    # FLIR camera simulator we're embedding in this code -> create_flir_img() method

    fire_img = False
  
    for i in range(width):
        for j in range(height):

            # TODO - This "for" body needs revision. The output composite image has 3 channels 
            # with the same info on each one (grayscale). It must be just one channel (cheaper). 
            # There is not need for casts. I should redefine the IF condition (If ITS WHITE), 
            # since we wont do different scales of heat anymore. DO IT WHILE THE ENV. BUILDING PROCESS 
            # TODO END 

            # getting the THERMAL pixel value.
            r, g, b = thermal_image.getpixel((i, j))

            # If the pixel is WHITE (#FFFFFF) then it's hot! -> Therefore we set the 255 value on the RGB image (scene)
            #
            if (int(r)==255 or int(g)==255 or int(b)==255):
                rgb_pixel_map[i, j] = (int(r), int(g), int(b))
                fire_img = True

            #If it's not, the we just turn the pixel on the RGB image to its grayscale equivalent
            else: 

                # getting the RGB pixel value.
                r, g, b = rgb_image.getpixel((i, j)) 

                # Apply formula of grayscale:
                grayscale = (0.299*r + 0.587*g + 0.114*b)

                # setting the pixel value.
                rgb_pixel_map[i, j] = (int(grayscale), int(grayscale), int(grayscale))

    # Saving the final output -- DEBUG -> pending to set a relative path 
    if (fire_img):
        rgb_image.save(composite_img_path, format="png")

def main(client,
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

        vector, angle, ir, scene = get_image(camera,
                                pose.position.x_val, 
                                pose.position.y_val, 
                                height, 
                                numpy.radians(pitch), 
                                roll, 
                                yaw, 
                                client)

        # Image class labeling - We use the dataset's directory tree structure to set 
        # the image class labeling depending on the UE4 environment zone we're 
        # taking the pictures of. We can implement this using an elif 
        # ladder, since there is no buit-in switch construct in Python 
        # -> https://pythongeeks.org/switch-in-python/

        def set_class_folder(input):
            if (ue4_zone == 0):
                selection = 'test/high-intensity-wildfires/'
            elif (ue4_zone == 1):
                selection = 'training+validation/high-intensity-wildfires/'
            elif (ue4_zone == 2): 
                selection = 'test/medium-intensity-wildfires/'
            elif (ue4_zone == 3):
                selection = 'training+validation/medium-intensity-wildfires/'
            elif (ue4_zone == 4): 
                selection = 'test/low-intensity-wildfires/'
            elif (ue4_zone == 5):
                selection = 'training+validation/low-intensity-wildfires/'
            # TODO elif (ue4_zone == 7):
            #     class_folder = 'no-wildfires/'
            return selection

        class_folder = set_class_folder(ue4_zone)

        # Adding positional metadata into the images filename and saving into the class folder

        thermal_img_path = (datasetFolder + 
                                class_folder + 
                                irFolder +                                                
                                'SEGMENT_'+ 
                                str(i).zfill(5) + '_' +
                                str('%.5f'%(pose.position.x_val)) + '_' +
                                str('%.5f'%(pose.position.y_val)) + '_' +
                                str(height) + '_' +
                                str(pitch) + '_' +
                                str(roll) + '_' +
                                str(yaw) +
                                '.png')
        
        rgb_img_path = (datasetFolder + 
                                class_folder + 
                                sceneFolder + 
                                'RGB_' +
                                str(i).zfill(5) + '_' +
                                str('%.5f'%(pose.position.x_val)) + '_' +
                                str('%.5f'%(pose.position.y_val)) + '_' +
                                str(height) + '_' +
                                str(pitch) + '_' +
                                str(roll) + '_' +
                                str(yaw) +
                                '.png')

        composite_img_path = (datasetFolder + 
                                class_folder + 
                                compositeFolder +
                                'FLIR_' +
                                str(i).zfill(5) + '_' +
                                str('%.5f'%(pose.position.x_val)) + '_' +
                                str('%.5f'%(pose.position.y_val)) + '_' +
                                str(height) + '_' +
                                str(pitch) + '_' +
                                str(roll) + '_' +
                                str(yaw) +
                                '.png')
        
        if writeIR:
            cv2.imwrite(thermal_img_path, ir)
        if writeScene:
            cv2.imwrite(rgb_img_path, scene)
        
        # with the "create_flir_img" method we combine both the RGB and SEGMENT 
        # images, obtaining the simulated FLIR thermal vision images as a result.
        
        create_flir_img(thermal_img_path,rgb_img_path,composite_img_path)
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
                "Zone 6 (Class 1+2+3: PoC experiments zone = 6\n",
                "Zone 7 (Class 4: images with no wildfires) = 7\n")
        ue4_zone = int(input("Please choose an option (0-7 - Default = 1): ") or '1')
        if ue4_zone==7:
            print("\nERROR: Class not implemented yet.\n")
            time.sleep(2)
        elif ue4_zone==6:
            print("\nERROR: Zone reserved to perfom the PoC experiments (so we avoid overfitting the model by memorizing features).\n")
            time.sleep(4)
        elif ue4_zone<0 or ue4_zone>7:
            print('\nERROR: Wrong option, try again.')
            time.sleep(2)
        else:
            wrong_option=False;

    wrong_option=True;
    while (wrong_option):
        print("Choose the multicopter's camera you want to use to retrieve the images:\n\n", 
                "front_center=0\n",
                "front_right=1\n",
                "front_left=2\n",
                "fpv=3\n",
                "back_center=4\n")
        camera = int(input("Please choose an option (0-4 - Default = 0):\n\n") or '0')
        if camera<0 or camera>4:
            print('\nERROR: Wrong option, try again.')
            time.sleep(2)
        else:
            wrong_option=False;

    height = int(input("Set the multicopter's height (negative integer value - Default = 20 -> lowest hight):\n\n") or '20')

    if height==-60:
        dataset = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/datasets/buffer/dataset#1_long-range-images/'
    elif height>-20 or height<100:
        dataset = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/datasets/buffer/dataset#2_close-range-images/'

    pitch = int(input("Set the camera's pitch angle (Integer degrees 180 > angle > 360 - Default = 270):\n\n") or '270')

    # Look for objects with names that match a regular expression. 
    # On the case of this PoC, we're looking for objects that include 
    # the "firewood" and the "grass" strings on the UE4 env. objects 
    # that simulate heat emission (see the project's report, section XXX).
    #
    # V4.6 -> Enabling the possiblity of generating only specific image classes 
    #         (see section 4.x.x of the project's report) by "injecting" the zone
    #         variable into the regex expression that filters the objects we want
    #         to take pictures of.

    my_regex1 = r".*?mesh_firewood_" + str(ue4_zone) +r".*?"
    my_regex2 = r".*?grass_mesh_" + str(ue4_zone) +r".*?"
    
    objectList = client.simListSceneObjects(my_regex1)
    objectList += client.simListSceneObjects(my_regex2)
    
    #Call to main
    main(client, 
         objectList,
         ue4_zone, 
         camera,
         height,
         pitch, 
         datasetFolder = dataset) 
