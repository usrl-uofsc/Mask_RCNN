import cv2
imagePath = '/home/michailk/projects/railway_detection/Code/MaskRCNN_Attempt_2/Mask_RCNN/dataset/masksAndImages/rs00504/images/rs00504.png'
image = cv2.imread(imagePath)
#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(image.shape)