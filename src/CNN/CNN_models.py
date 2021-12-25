import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

###################################################################################################
###################################################################################################
####                                                                                           ####   
####                          1. CNN MODEL VERSION #1 (3 layers architecture)                  ####
####                                                                                           ####   
###################################################################################################
###################################################################################################

# 2.1 - First we define a pretty simple, 2-layer convolution neural network in order to guess the
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
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

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output