import setup_path
import airsim

import sys
import time

print("""This script HAS BEEN ADAPTED FROM path.py""")
print("""https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo""")


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

# AirSim uses NED coordinates so negative axis is up.
# z of -5 is 5 meters above the original launch point.

### TFG - Suposem (millor dit, imaginem de cara a la PoC) que l'edge-device incorporat al dron 
### diposa d'una cartografia digitalitzada del terreny, de dades sobre l'alçada màxima dels 
### àrbres i d'un sistema de detecció d'objectes. Per tant, establint z = -5 sobre el terreny 
### estàtic (sempre el mateix) ja evitem colisions amb cap objecte i podem simular aquest escenari.
 
z = -5
print("make sure we are hovering at {} meters...".format(-z))
client.moveToZAsync(z, 1).join()

# see https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo

# this method is async and we are not waiting for the result since we are passing timeout_sec=0.

##DEBUG
##
print("flying on path...")
result = client.moveOnPathAsync([airsim.Vector3r(-200,200,75)],
                        5, 200,
                        airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()
'''
while (0):                        
    print("flying on circles forever...")
    result = client.moveOnPathAsync([airsim.Vector3r(10,0,0),
                                    airsim.Vector3r(0,-10,0),
                                    airsim.Vector3r(-10,0,0),
                                    airsim.Vector3r(0,10,0)],
                            5, 200,
                            airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()


### TFG - Procedim a realitzar captures d'imatges cada 5 segons i utilitzant totes les càmeres que ja 
### proporciona el multirotor (buit-in):
#while result == False:
responses = client.simGetImages([
airsim.ImageRequest(0, airsim.ImageType.DepthVis),
airsim.ImageRequest(1, airsim.ImageType.Infrared),
airsim.ImageRequest(2, airsim.ImageType.DepthPlanar, True)])
print('Retrieved images:', len(responses))
'''

# drone will over-shoot so we bring it back to the start point before landing.
client.moveToPositionAsync(0,0,z,1).join()
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")