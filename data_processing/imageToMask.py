from cv2 import COLOR_RGB2GRAY
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

jpgDirectory = '/home/michailk/projects/railway_detection/Data/RailSem19/test/jpgs/'
uint8Directory = '/home/michailk/projects/railway_detection/Data/RailSem19/test/uint8/'
finalDirectory = '/home/michailk/projects/railway_detection/Code/MaskRCNN_Attempt_2/Mask_RCNN/dataset/masksAndImages/'
fullMaskDirectory = '/home/michailk/projects/railway_detection/Data/RailSem19/masks/'
for filename in os.listdir(jpgDirectory):
#for filename in os.listdir(uint8Directory):
    uint8File = os.path.splitext(os.path.join(uint8Directory, filename))[0]+'.png'
    #uint8File = os.path.join(uint8Directory, filename)
    img = cv2.imread(uint8File)
    jpgFile = os.path.join(jpgDirectory, filename) 
    #jpgFile = os.path.splitext(os.path.join(jpgDirectory, filename))[0]+'.jpg'
    original = cv2.imread(jpgFile)

    a, railsBottomThresh  = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY) 
    a, railsTopThresh = cv2.threshold(img, 17, 255, cv2.THRESH_BINARY) 
    rails = railsBottomThresh - railsTopThresh

    a, trackBottomThresh  = cv2.threshold(img, 11, 255, cv2.THRESH_BINARY) 
    a, trackTopThresh = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY) 
    track = trackBottomThresh - trackTopThresh

    trackAndRails = rails + track

    trackAndRails = cv2.cvtColor(trackAndRails, cv2.COLOR_RGB2GRAY)
    original = cv2.cvtColor(original, COLOR_RGB2GRAY)

    #cv2.imshow('Unfiltered', img)
    #cv2.imshow('Filtered', newImg)
    #cv2.imshow('Original', original)
    #cv2.imshow('Track', trackBlur)
    #cv2.imshow('Rails', rails)
    #cv2.imshow('Tracks and Rails', trackAndRails)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imageName = os.path.splitext(filename)[0]
    
    os.mkdir('{}{}/'.format(finalDirectory, imageName))
    os.mkdir('{}{}/images/'.format(finalDirectory, imageName))
    os.mkdir('{}{}/masks/'.format(finalDirectory, imageName))

    originalFinalPath = '{}{}/images/{}.png'.format(finalDirectory, imageName, imageName)
    cv2.imwrite(originalFinalPath, original)

    trackAndRailFinalPath = '{}{}/masks/{}.png'.format(finalDirectory, imageName, imageName)
    cv2.imwrite(trackAndRailFinalPath, trackAndRails)

    maskPath = '{}{}.png'.format(fullMaskDirectory, imageName)
    cv2.imwrite(maskPath, trackAndRails)
    


