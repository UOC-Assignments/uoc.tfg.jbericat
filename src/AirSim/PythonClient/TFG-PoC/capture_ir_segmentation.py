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
   
def get_image(x, y, z, pitch, roll, yaw, client):
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

    #Set pose and sleep after to ensure the pose sticks before capturing image.
    client.simSetVehiclePose(Pose(Vector3r(x, y, z), \
                      to_quaternion(pitch, roll, yaw)), True)
    time.sleep(0.1)

    #Capture segmentation (IR) and scene images.
    responses = \
        client.simGetImages([ImageRequest("0", ImageType.Infrared,
                                          False, False),
                            ImageRequest("0", ImageType.Scene, \
                                          False, False),
                            ImageRequest("0", ImageType.Segmentation, \
                                          False, False)])

    #Change images into numpy arrays.

   ## S'HA HAGUT DE MODIFICAR EL CODI PER A QUE COMPILI!! (el darrer paràmetre de .reshape ha de ser 3 i no 4, falta validar-ho ) 
    img1d = numpy.fromstring(responses[0].image_data_uint8, dtype=numpy.uint8)
    im = img1d.reshape(responses[0].height, responses[0].width, 3) 

    img1dscene = numpy.fromstring(responses[1].image_data_uint8, dtype=numpy.uint8)
    imScene = img1dscene.reshape(responses[1].height, responses[1].width, 3)

    return Vector3r(x, y, z), to_quaternion(pitch, roll, yaw),\
           im[:,:,:3], imScene[:,:,:3] #get rid of alpha channel

    
def combine_img(thermal_img_path, rgb_img_path, composite_img_path):
    """
    title::
        combine_img

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
  
    for i in range(width):
        for j in range(height):

            # getting the THERMAL pixel value.
            r, g, b = thermal_image.getpixel((i, j))

            # If the pixel is not BLACK (0,0,0) -> then it is grayscaled -> 
            # -> then it's hot! -> Therefore we set its value on the RGB image (scene)
            if (int(r)!=0 or int(g)!=0 or int(b)!=0):
                rgb_pixel_map[i, j] = (int(r), int(g), int(b))

            #If it's not, the we just turn the pixel to its grayscale equivalent
            else: 

                # getting the RGB pixel value.
                r, g, b = rgb_image.getpixel((i, j)) 

                # Apply formula of grayscale:
                grayscale = (0.299*r + 0.587*g + 0.114*b)

                # setting the pixel value.
                rgb_pixel_map[i, j] = (int(grayscale), int(grayscale), int(grayscale))

    # Saving the final output -- DEBUG -> pending to set a relative path 
    rgb_image.save(composite_img_path, format="png")

def main(client,
         objectList,
         pitch=numpy.radians(270), #image straight down
         roll=0,
         yaw=0,
         z=-122,
         writeIR=True,
         writeScene=True,
         irFolder='',
         sceneFolder='',
         compositeFolder=''):
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

    author::
        Elizabeth Bondi
    """
    i = 0
    for o in objectList:
        pose = client.simGetObjectPose(o);
        
        #Capture image - pose.position x_val access may change w/ AirSim
        #version (pose.position.x_val new, pose.position[b'x_val'] old)
        """
        vector, angle, ir, scene = get_image(pose.position.x_val, 
                                                100, 
                                                z, 
                                                6, 
                                                roll, 
                                                5, 
                                                client)
        """
        vector, angle, ir, scene = get_image(pose.position.x_val, 
                                                pose.position.y_val+10, 
                                                pose.position.z_val, 
                                                6, 
                                                roll, 
                                                5, 
                                                client)
        #Convert color scene image to BGR for write out with cv2.
        r,g,b = cv2.split(scene)
        scene = cv2.merge((b,g,r))

        thermal_img_path = irFolder+'ir_'+str(i).zfill(5)+'.png'
        rgb_img_path = sceneFolder+'scene_'+str(i).zfill(5)+'.png'
        composite_img_path = compositeFolder+'composite_'+str(i).zfill(5)+'.png'

        if writeIR:
            cv2.imwrite(thermal_img_path, ir)
        if writeScene:
            cv2.imwrite(rgb_img_path, scene)
        
        #DEBUG: funció "combine_img" TFG Upgrade: 
        combine_img(thermal_img_path,rgb_img_path,composite_img_path)

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

    #Look for objects with names that match a regular expression.
    #landList = client.simListSceneObjects('.*?Landscape.*?')
    fireList = client.simListSceneObjects('.*?Fire.*?')
    #treeList = client.simListSceneObjects('.*?Tree.*?')
    
    objectList = fireList
    
    #Sample calls to main
    main(client, 
         objectList, 
         irFolder='/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/datasets/buffer/IR/',
         sceneFolder='/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/datasets/buffer/scene/',
         compositeFolder='/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/datasets/buffer/composite/') 
