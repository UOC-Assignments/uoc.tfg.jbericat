"""
Title::

    drone_patrol.py 

Python environment: 

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


#import setup_path
import airsim
import sys
import time

'''This script HAS BEEN ADAPTED FROM path.py
'''

ZONE6A_NE_X = 34616.03125
ZONE6A_NE_Y = -47017.53125
ZONE6A_NE_Z = 19651.738281

ZONE6A_NW_X = -44855.246094
ZONE6A_NW_Y = -83644.671875
ZONE6A_NW_Z = 1948.820312

ZONE6A_SE_X = -16046.804688
ZONE6A_SE_Y = -56172.183594
ZONE6A_SE_Z = -5232.59668

ZONE6A_SW_X = -55523.9375
ZONE6A_SW_Y = -59685.925781
ZONE6A_SW_Z = -2311.015625

'''ZONE6B_NE_X = 
ZONE6B_NE_Y = 
ZONE6B_NE_Z = 

ZONE6B_NW_X = 
ZONE6B_NW_Y = 
ZONE6B_NW_Z = 

ZONE6B_SE_X = 
ZONE6B_SE_Y = 
ZONE6B_SE_Z = 

ZONE6B_SW_X = 
ZONE6B_SW_Y = 
ZONE6B_SW_Z = '''

client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)

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


# Starting with ZONE SIX (Low-Intensity Wildfires):
#client.moveToPositionAsync(x=ZONE6_SE_X, y=ZONE6_SE_Y, z=ZONE6_SE_Z, velocity=40)
#client.moveToPositionAsync(x=ZONE6A_NW_X, y=ZONE6A_NW_Y, z=ZONE6A_NW_Z, velocity=60) 
#client.moveToPositionAsync(x=ZONE6A_SW_X, y=ZONE6A_SW_Y, z=ZONE6A_SW_Z, velocity=5)
#client.moveToPositionAsync(x=ZONE6A_NE_X, y=ZONE6A_NE_Y, z=ZONE6A_NE_Z, velocity=5) # this is the PLAYER_START position

print("flying on path...")


result = client.moveToPositionAsync(x=ZONE6A_SW_X, y=ZONE6A_SW_Y, z=ZONE6A_SW_Z, velocity=40).join()
print(result)

'''result = client.moveOnPathAsync([airsim.Vector3r(-200,200,75)],
                        10, 200,
                        airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()
'''


'''
##DEBUG


            
print("flying on circles forever...")
result = client.moveOnPathAsync([airsim.Vector3r(400,0,0),
                                airsim.Vector3r(0,-400,0),
                                airsim.Vector3r(-400,0,0),
                                airsim.Vector3r(0,400,0)],
                        5, 5,
                        airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()
print(result)

'''

# drone will over-shoot so we bring it back to the start point before landing.
# client.moveToPositionAsync(0,0,z,1).join()

# TODO Here we need to implement a sync mechanism....
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")