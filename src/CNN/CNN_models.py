"""
title:: 


description::


inputs::


output::


author::
    Version 1 & 2: Jordi Bericat Ruz - Universitat Oberta de Catalunya
    Version 3: Microsoft Docs - Windows AI - Windows Machine Learning

references::
    [1] - Microsoft Docs - Windows AI - Windows Machine Learning -> https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
    [2] - PyTorch - Python Deep Learning Neural Network API -> https://deeplizard.com/learn/video/IKOHHItzukk
    [3] - Image Kernels Explained Visually -> https://setosa.io/ev/image-kernels/
"""

import torch.nn as nn
import torch.nn.functional as F

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                          1. CNN MODEL VERSION #1 (3 layers architecture)                  ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

# First we define a pretty simple, 2-layer convolution neural network in order to guess the
# best hyper-parameter configuration possible - That is, the one that works with the given data
# (229 x 229 x 1 images) as well as provides with best accuracy results:

class Network_v1(nn.Module):
     
    def __init__(self):
        # By inheriting from thorch.nn's Network() class, we are able to use it's extended features in our 
        # own class definition (such as printing the model architecure) wehen invoking the class constructor
        super(Network_v1, self).__init__()

        # Next we set the convolutional, pass-through (batch norm, pooling) and output layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(229*229*16, 3) 

# 2.2 - Implement the forward function:
    def forward(self, input):
        output = F.relu(self.conv1(input))      
        output = F.relu(self.conv2(output))        
        output = output.view(-1, 229*229*16 ) 
        output = self.fc1(output)

        return output


###################################################################################################
###################################################################################################
####                                                                                           ####
####                          2. CNN MODEL VERSION #2 (9 layers architecture)                  ####
####                                                                                           ####
###################################################################################################
###################################################################################################


class Network_v2(nn.Module):
     
    def __init__(self):

        super(Network_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=3, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(114*114*32 , 3) 

# 2.2 - Implement the forward function:
    def forward(self, input):
   
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.pool1(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool2(output)
        output = output.view( -1, 114*114*32)
        output = self.fc1(output)

        return output

###################################################################################################
###################################################################################################
####                                                                                           ####
####                          3. CNN MODEL VERSION #3 (14 layers architecture)                 ####
####                                                                                           ####
###################################################################################################
###################################################################################################

# See this project report's (section 5.2.3.3) for further information in regards to the calculations
# made to set the output-channel size (linear layer)

# Define a convolution neural network
class Network_v3(nn.Module):
    def __init__(self):
        super(Network_v3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*58*58, 3)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(-1, 32*58*58)
        output = self.fc1(output)

        return output