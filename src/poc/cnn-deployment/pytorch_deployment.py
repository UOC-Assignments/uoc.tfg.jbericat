"""
Title::

    pytorch_deployment_PoC#1_v1.0-.py 

Python environment: 

    TODO Python 3.6.4 + pytorch + CUDA, etc....

Description::

    Deploying the classification algorithm (model cnn-training_v3.pth) WITHOUT
    localization boxes (just image classification).
    
    IMPORTANT NOTICE: NO REAL-TIME PROCESSING ON THIS PoC VERSION (PoC#1_v1.0). 
    
    Image analysis is done OFFLINE once the image collection by the drone's is
    finished. This is a WORK-IN-PROGRESS. The goal is to process images on 
    realtime. However, to do so I should find a way to do the create_flir_img()
    function computation in a way more efficient way. By instance, using cuda 
    would allow me to scan the 512 rows of every SEGMENT image AT ONCE (GPU has
    896 cores, so it should suffice). Another way would be just cpu-threading 
    the sequential process (IDK how efficiently I can do that with python, 
    though). 

Inputs::

    1. RAW .png images (SEGMENT-IR & SCENE-RGB) taken by the drone during the 
       surveillance flight (airsim_drone_survey.py) -> /usr/PoC/in/%sample_folder%/

    2. Trained CNN Model .pth file -> /usr/PoC/CNN/trained-model.pth

Output::

    1. Unlabelel Image-grid summary itended for visual inspection 
       -> /usr/PoC/out/labels-prediction.png

    2. STODUT redirected to file (table with predictions ordered by frame number) 
       -> /usr/PoC/out/frame-predictions.png 


Original author::

    Jordi Bericat Ruz - Universitat Oberta de Catalunya

References::

    1 - https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#MultirotorClient.moveOnPathAsync 
    2 - https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo 
    3 - https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html


TODO list: 
    1 - "Realtimize" the whole process: It would require the implementation of
        syncronization mechanisms (e.g. a queue to retrieve the data from "airsim_rec.txt", mutex or locks for the frame_count, etc)
    2 - Try catch clause for when the IN data folder is empty

"""

# IMPORTS

## PART I 

import cv2 as cv # https://stackoverflow.com/questions/50909569/unable-to-import-cv2-module-python-3-6
import os

from tabulate import tabulate

## PART II 

# DEBUG: We copied some code (defs) into this project folder in a rush. Create a shared lib in /usr/lib by instance....

# Importing functions from the PoC Library folder /src/poc/lib
import sys 
sys.path.insert(0, '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/') # This one is the git src folder 

from poc.lib.CNN_models import *

import numpy as np
import sys
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.functional import Tensor
import matplotlib.pyplot as plt

# Base folders
POC_FOLDER = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/' 
FLIR_BUFFER = POC_FOLDER + 'flir_buffer/'
PREDICTIONS_OUT_FILE = open(os.path.abspath(POC_FOLDER + 'out/frame-predictions.log'), "w")   

#UE4_ZONE_6 = 6
#UE4_ZONE_7 = 7

GREEN_COLOR = [0,255,0]
YELLOW_COLOR = [0,255,255]
RED_COLOR = [0,0,255]

#data-bond constants
DEPLOY_DATA_DIR = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer"
CNN_IMG_SIZE = 229

#model-bond constants
MODEL_VERSION = 3
MODEL_PATH = "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/"

# To simulate a GPU device embeded into the drone's companion computer (e.g. NVIDIA Jetson Nano)
# now we're going to process the images one by one. So we won't be using batch sizes
BATCH_SIZE = 128

def add_bounding_box(prediction):
    #TODO DOC - https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    borderType = cv.BORDER_CONSTANT

    # Load an image
    # TODO - Path structure is a bit of a mess...
    frame_path = FLIR_BUFFER + 'unknown-class/frame-' + str(prediction[0]) + '.png'
    src = cv.imread(cv.samples.findFile(frame_path), cv.IMREAD_COLOR)

    # Check if the image was correctly loaded 
    if src is None:
        print ('Error opening image!')
        return -1

    
    #TODO DOC
    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * src.shape[1])  # shape[1] = cols
    right = left

    if prediction[1] == 'no-wildfires':
        #add_green_boundingbox
        color = GREEN_COLOR

    elif prediction[1] == 'low-intensity-wildfires':
        #add_yellow_boundingbox
        color = YELLOW_COLOR

    elif prediction[1] == 'high-intensity-wildfires':
        #add_green_boundingbox
        color = RED_COLOR

    #TODO DOC
    dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, color)
   
    # SAVE TO FILE (OVERWRITING THE BUFFER IS OK)
    cv.imwrite(frame_path, dst)


# PART II DEFINITIONS

def poc_one_deploy_BAK():

    '''TODO DOCU - Function to test the model with a batch of images and show the labels predictions'''

    # stdout
    print("\n***************************************************************************************************************************\n" + 
          " CNN Deployment - PoC #1: Image classification WITHOUT localization (That is, adding bounding boxes to the captured images)" +
          "\n***************************************************************************************************************************\n")

    #######################################################################
    # STEP 1: Loading the deployment data 
    #######################################################################

    # Define transformations for the training and test subsets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        # Normalizing the images ___________
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(1), # DEBUG -> THIS IS A WORKAROUND; IMAGES ARE EXPECTED TO BE OF ONE CHANNEL ONLY (GRAYSCALE)
        # We need square images to feed the model (the raw dataset has 640x512 size images)
        # DEBUG - UNCOMMENT NEXT LINE FOR v4 and v9 DATASETS
        #transforms.RandomResizedCrop(512),
        # Now we just resize into any of the common input layer sizes (32×32, 64×64, 96×96, 224×224, 227×227, and 229×229)
        transforms.Resize(CNN_IMG_SIZE)
    ])

    # Create an instance for deployment
    deploy_data = (datasets.ImageFolder(root=DEPLOY_DATA_DIR, transform=transformations))
    
    # Create a loader for the test set which will read the data within batch size and put into memory. 
    # Note that each shuffle is set to false, since we will be creating a frame-by-frame animation 
    deploy_loader = DataLoader(deploy_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # DEBUG - batch_size=len(deploy_data)

    # TODO -> DOCU
    classes = ('high-intensity-wildfires', 'low-intensity-wildfires', 'no-wildfires')

    # TODO -> DOCU
    print( " - The number of images in a deploy set is: " + str(len(deploy_data)) + "\n" )
    print( " - The batch-size is: " + str(BATCH_SIZE) + "\n" )
    print( " - The number of batches needed to procees the whole deployment data is:" + str(int(len(deploy_data)//BATCH_SIZE)) + "\n\n" ) # DEBUG - There is a first grade's misscalculation here ^^' use mod op (%) or whatever suits better

    #######################################################################
    # STEP 2: Loading the CNN model and importing the data  
    #######################################################################

    # Let's load the model that got best accuracy during training (86%) for 
    # the PoC-ONE dataset:

    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - Model deployed on", device, "device")

 
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()


    # DEBUG - THE LOOP STARTS HERE
    # for j in range(len(deploy_data)):
    mydata = []
    frame_id = 0 #COUNTER

    for i, (images, labels) in enumerate(deploy_loader, 0):
    
        # get batch of images from the test DataLoader  
        images, labels = next(iter(deploy_loader)) 

        print("BATCH NUMBER",i)
        
        # MOVING DATA TO THE GPU MEM SPACE
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # show all images as one image grid
        img_grid = Tensor.cpu(torchvision.utils.make_grid(images))
        img_grid = img_grid / 2 + 0.5  # unnormalize
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(POC_FOLDER+'out/labels-prediction.png')
        plt.show()

        #######################################################################
        # STEP 3: Peforming predictions over the imported data  
        #######################################################################
        
        # Let's see what if the model identifies the labels of these example
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        #######################################################################
        # STEP 4: Showing results  
        #######################################################################

        # Even though we can't process 

        # Preparing the data to build a table 

        for j in range(len(outputs)):
            # On this implementation (REAL CASE SIMULATION) we actually don't know what
            # the object classification is, so it must be visually checked afterwards. 
            # However, we can be sure that the accuracy of the prediction will be of the 86% 
            # with a small variation index (the model accuracy is pretty stable, 
            # as seen in the training stats on the section 6.x.x.x). THEREFORE, 
            # we don't add class labels to the images extracted from the deployment
            # dataset (labeled as unkown-class on the deployment dataset). Instead, 
            # we'll be labeling it with the frame/image sequential id (adding the filename 
            # would be ideal but there is no more time for fancy stuff). This way, an operator 
            # could verify the validity of a small sample, but big enough to be of significance.

            # assign data
            predicted_label = classes[predicted[j]]
            print("(frame_id,i,j,prediction)",frame_id,i,j,predicted_label)
            mydata.append([frame_id, predicted_label])
            frame_id += 1
    
    # create header
    head = ["Frame ID", "Classification Result"]
    
    # print the table to info file
    print(tabulate(mydata, headers=head, tablefmt="grid"), file=PREDICTIONS_OUT_FILE)

    for i in range(len(deploy_data)): # TODO DEBUG - THIS IS A PATCH! We're creating a myData list of size multiple of the batch-size, instead of the deploy data size
        # TODO - DOC
        add_bounding_box(mydata[i])

import torch.onnx 
import torch
from torchvision import transforms
from PIL import Image

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - ONNX Coversion performed on", device, "device")
 
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # Setting evalatuion mode, since we don't want to retrain the model
    model.eval()

    # Let's load an image and convert it to tensor datatype  
    #img = Image.open("/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/flir_buffer/unknown-class/frame-89.png")
    #convert_tensor = transforms.ToTensor()
    dummy_input = torch.randn(BATCH_SIZE, 1, 229, 229, device="cuda") # TODO DEBUG - WHAT's WITH THE FIRST ARGUMENT?

    #We also compute torch_out, the output after of the model, which we will use to verify that the model we exported computes the same values when run in ONNX Runtime.
    torch_out = model(dummy_input)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/PoC/CNN/ImageClassifier.onnx",       # where to save the model  
         verbose=True,
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')


    ######

    import onnxruntime as rt

    ort_session = rt.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


    ############################################################################
    ###############################      PART II     ###########################
    ############################################################################
    ##
    ##  PoC #1: Image classification WITHOUT adding bounding boxes to the FLIR images, 
    ##          which contain wilfire instances from multiples classes
    ##
    ##  https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
    ##  https://medium.com/analytics-vidhya/guide-to-object-detection-using-pytorch-3925e29737b9 
    ##
    ###########################################################################


# CALLING MAIN.
if __name__ == '__main__':
    model = set_model_version(MODEL_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH + "trained-model.pth"))
    
    Convert_ONNX() 



'''
'''
