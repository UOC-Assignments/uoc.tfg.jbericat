"""
Title::

    pytorch_deployment_PoC#1_v1.0-.py 

Conda environment: 

    "conda activate condapy373"
    
    Python 3.7.3 + pytorch + CUDA, etc....

Description::

Inputs::

Output::

Original author::

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - 

TODO list: 
    1 - 
"""

# IMPORTS

## PART I 

from genericpath import isfile
from posixpath import join

from numpy.core.fromnumeric import take

# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder

import poc.lib.flir_simulator as flir

#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
import sys

# CALLING MAIN.
if __name__ == '__main__':

    ###########################################################################
    ###############################      PART I      ##########################
    ###########################################################################
    ##
    ##      Simulating FLIR camera video capture (0.5 FPS) in real-time  
    ##
    ###########################################################################

    # Let's run the night/thermal-vision simulation on the images taken by the drone on it's LATEST flight
    flir.offline_batch_coverter()