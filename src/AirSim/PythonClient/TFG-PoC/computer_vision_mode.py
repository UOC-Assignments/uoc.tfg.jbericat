# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path
import airsim

import pprint
import os
import time
import math
import tempfile

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

airsim.wait_key('Press any key to set camera-0 gimbal to 15-degree pitch')
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0)) #radians
client.simSetCameraPose("0", camera_pose)

airsim.wait_key('Press any key to get camera parameters')
for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %d:" % camera_name)
    pp.pprint(camera_info)

tempfile.tempdir = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/lib/images/raw/'
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

airsim.wait_key('Press any key to get images')

pose = client.simGetVehiclePose()

print ("Choose the multicopter camera you want to use to retrieve the images:\n\n", 
    "front_center=0\n",
    "front_right=1\n",
    "front_left=2\n",
    "fpv=3\n",
    "back_center=4\n"
    )

camera = input("Please enter a number (0-4):\n")

print ("Choose the image type:\n\n",
    "Scene = 0\n", 
    "DepthPlanar = 1\n",
    "DepthPerspective = 2\n",
    "DepthVis = 3\n", 
    "DisparityNormalized = 4\n",
    "Segmentation = 5\n",
    "SurfaceNormals = 6\n",
    "Infrared = 7\n",
    "Thermal(Not implemented yet) = 8\n"
    )

img_type = input("Please enter a number (0-8):\n")

for x in range(10): # do few times
    print ("Current position (x component): %d" % pose.position.x_val)
    print ("Current position (y component): %d" % pose.position.y_val) 
    print ("Current position (z component): %d" % pose.position.z_val) 
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pose.position.x_val, pose.position.y_val, pose.position.z_val)), True)

    responses = client.simGetImages([
       # airsim.ImageRequest(camera, img_type),
        airsim.ImageRequest(camera, int(img_type))],
        )

    for i, response in enumerate(responses):
        filename = os.path.join(tmp_dir, str(x) + "_" + str(i))
        if response.pixels_as_float:
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)

    pose.position.x_val += 10
    time.sleep(1)

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)