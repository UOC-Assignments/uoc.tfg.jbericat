"""
Title::

    drone_patrol.py 

Conda environment: 

    conda activate condapy373
    
    TODO Python 3.7.3 + AirSim x.y....... 

Description::

    Implementing a Drone Survey around the two PoC zones (Zone 6A and Zone 6B)

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

# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 

from poc.lib.setup_path import *
from poc.lib.airsim_set_environment import *
import poc.lib.flir_simulator as flir

import airsim
import sys
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)


# SET custom StaticMeshObjects thermal emitters
set_environment(client)
#time.sleep(5)

print("arming the drone...")
client.armDisarm(True)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    client.hoverAsync().join()

time.sleep(1)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("take off failed...")
    sys.exit(1)

ZONE_6A_NE_X = client.simGetObjectPose("ZONE_6A_NE").position.x_val;
ZONE_6A_NE_Y = client.simGetObjectPose("ZONE_6A_NE").position.y_val;
ZONE_6A_NE_Z = client.simGetObjectPose("ZONE_6A_NE").position.z_val;

ZONE_6A_NW_X = client.simGetObjectPose("ZONE_6A_NW").position.x_val;
ZONE_6A_NW_Y = client.simGetObjectPose("ZONE_6A_NW").position.y_val;
ZONE_6A_NW_Z = client.simGetObjectPose("ZONE_6A_NW").position.z_val;

ZONE_6A_SE_X = client.simGetObjectPose("ZONE_6A_SE").position.x_val;
ZONE_6A_SE_Y = client.simGetObjectPose("ZONE_6A_SE").position.y_val;
ZONE_6A_SE_Z = client.simGetObjectPose("ZONE_6A_SE").position.z_val;

ZONE_6A_SW_X = client.simGetObjectPose("ZONE_6A_SW").position.x_val;
ZONE_6A_SW_Y = client.simGetObjectPose("ZONE_6A_SW").position.y_val;
ZONE_6A_SW_Z = client.simGetObjectPose("ZONE_6A_SW").position.z_val;

ZONE_6B_NE_X = client.simGetObjectPose("ZONE_6B_NE").position.x_val;
ZONE_6B_NE_Y = client.simGetObjectPose("ZONE_6B_NE").position.y_val;
ZONE_6B_NE_Z = client.simGetObjectPose("ZONE_6B_NE").position.z_val;

ZONE_6B_NW_X = client.simGetObjectPose("ZONE_6B_NW").position.x_val;
ZONE_6B_NW_Y = client.simGetObjectPose("ZONE_6B_NW").position.y_val;
ZONE_6B_NW_Z = client.simGetObjectPose("ZONE_6B_NW").position.z_val;

ZONE_6B_SE_X = client.simGetObjectPose("ZONE_6B_SE").position.x_val;
ZONE_6B_SE_Y = client.simGetObjectPose("ZONE_6B_SE").position.y_val;
ZONE_6B_SE_Z = client.simGetObjectPose("ZONE_6B_SE").position.z_val;

ZONE_6B_SW_X = client.simGetObjectPose("ZONE_6B_SW").position.x_val;
ZONE_6B_SW_Y = client.simGetObjectPose("ZONE_6B_SW").position.y_val;
ZONE_6B_SW_Z = client.simGetObjectPose("ZONE_6B_SW").position.z_val;

### TFG - Suposem (millor dit, imaginem de cara a la PoC) que l'edge-device incorporat al dron 
### diposa d'una cartografia digitalitzada del terreny, de dades sobre l'alçada màxima dels 
### àrbres i d'un sistema de detecció d'objectes. Per tant, establint z = -5 sobre el terreny 
### estàtic (sempre el mateix) ja evitem colisions amb cap objecte i podem simular aquest escenari.

# AirSim uses NED coordinates so negative axis is up.
# z of -5 is 5 meters above the original launch point.

z = -10
print("make sure we are hovering at {} meters...".format(-z))
response = client.moveToZAsync(z, 1).join()
print(response)

# Starting with the ZONE-6A (High-Intensity Wildfires) at the North-East (NE) corner
# TODO - Those results are intended to be cached by a try-catch clause...
print("flying-through to the ZONE-6A SW corner...")
result = client.moveToPositionAsync(x=ZONE_6A_SW_X, y=ZONE_6A_SW_Y, z=ZONE_6A_SW_Z-10, velocity=10).join()
print("moveToPositionAsync result: " + str(result))

print("flying the perimeter to the ZONE-6A NW corner...")
result = client.moveToPositionAsync(x=ZONE_6A_NW_X, y=ZONE_6A_NW_Y, z=ZONE_6A_NW_Z-20, velocity=10).join()
print("moveToPositionAsync result: " + str(result))

print("flying-through again to the ZONE-6A SE corner...")
result = client.moveToPositionAsync(x=ZONE_6A_SE_X, y=ZONE_6A_SE_Y, z=ZONE_6A_SE_Z-20, velocity=10).join()

# Gaining some height...
client.moveToZAsync(-5, 4).join()

# Jumping to ZONE-6B Now...

print("flying-through to the ZONE-6B SW corner...")
client.moveToPositionAsync(x=ZONE_6B_SW_X, y=ZONE_6B_SW_Y, z=ZONE_6B_SW_Z-10, velocity=30).join()

print("flying-through to the ZONE-6B NE corner...")
client.moveToPositionAsync(x=ZONE_6B_NE_X, y=ZONE_6B_NE_Y, z=ZONE_6B_NE_Z-10, velocity=10).join()

print("flying-through to the ZONE-6B SE corner...")
client.moveToPositionAsync(x=ZONE_6B_SE_X, y=ZONE_6B_SE_Y, z=ZONE_6B_SE_Z-10, velocity=10).join()

# Gaining some height again...
client.moveToZAsync(-5, 4).join()

print("flying-through to the ZONE-6B NW corner...")
client.moveToPositionAsync(x=ZONE_6B_NW_X, y=ZONE_6B_NW_Y, z=ZONE_6B_NW_Z-20, velocity=10).join()


# We won't bring back the drone, so the survey just ends here.

print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("Drone survey is done. Creating FLIR images now...")
# Let's run the night/thermal-vision simulation on the images taken by the drone on it's LATEST flight
flir.offline_batch_coverter()
print("All images for deplyment are on -> /home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer/")
