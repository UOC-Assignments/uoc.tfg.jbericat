# TEST CODE: Checking the dataset images's number of channels

import cv2

gray_img_path = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/raw_datasets/v5.0/test/high-intensity-wildfires/FLIR/FLIR_00000_206.09999_249.29999_-15_270_0_0.png'
rgb_img_path = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/raw_datasets/v5.0/test/high-intensity-wildfires/RGB/RGB_00000_206.09999_249.29999_-15_270_0_0.png'

image = cv2.imread(rgb_img_path, cv2.IMREAD_UNCHANGED)
print(image.shape)