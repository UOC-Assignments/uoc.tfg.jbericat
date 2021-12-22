
import cv2

img_path1 = '/home/jbericat/Workspaces/uoc.tfg.jbericat/src/CNN/data/training+validation/high-intensity-wildfires/FLIR_00000_-996.89996_-435.19998_-20_315_0_0.png'

img_path2 = '/home/jbericat/Workspaces/uoc.tfg.jbericat/usr/raw_datasets/v3.0_(3-channels-whitescale-images)/training+validation/high-intensity-wildfires/FLIR/FLIR_00000_-996.89996_-435.19998_-20_270_0_0.png'

image = cv2.imread(img_path1)

#height, width, channels = image.shape

len(image.shape)

print(len(image.shape))