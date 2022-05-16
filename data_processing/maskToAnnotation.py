# https://towardsdatascience.com/how-to-automatically-generate-vgg-image-annotation-files-41d226e6d85

import os
import cv2
from tqdm import tqdm
import json 

# def Annot(preprocess_path, res=0.6, surf=100, eps=0.01):
#     """
#     preprocess_path : path to folder with gt and images subfolder containing ground truth binary label image and visual imagery
#     res : spatial resolution of images in meter 
#     surf : the minimum surface of roof considered in the study in square meter
#     eps : the index for Ramer–Douglas–Peucker (RDP) algorithm for contours approx to decrease nb of points describing a contours
#     """
res=0.6
surf=100
eps=0.01
preprocess_path = '/home/michailk/projects/railway_detection/Code/MaskRCNN_Attempt_2/Mask_RCNN/dataset/stage1_train/'  
jsonf = {} # only one big annotation file
with open(os.path.join(preprocess_path,'via_region_data.json'), 'w') as js_file:
    gt_path = os.path.join(preprocess_path, 'gt')
    images_path = os.path.join(preprocess_path, 'images')
    
    # All the elements in the images folders
    lst = os.listdir(images_path)
    for elt in tqdm(lst, desc='lst'):
        
        # Read the binary mask, and find the contours associated
        gray = cv2.imread(os.path.join(gt_path, elt))
        imgray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
        # https://www.pyimagesearch.com/2021/10/06/opencv-contour-approximation/
        # Contours approximation based on Ramer–Douglas–Peucker (RDP) algorithm
        areas = [cv2.contourArea(contours[idx])*res*res for idx in range(len(contours))]
        large_contour = []
        for i in range(len(areas)):
            if areas[i]>surf:
                large_contour.append(contours[i])
        approx_contour = [cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True) for c in large_contour]
            
        # -------------------------------------------------------------------------------
        # BUILDING VGG ANNTOTATION TOOL ANNOTATIONS LIKE 
        if len(approx_contour) > 0:
            regions = [0 for i in range(len(approx_contour))]
            for i in range(len(approx_contour)):
                shape_attributes = {}
                region_attributes = {}
                region_attributes['class'] = 'track'
                regionsi = {}
                shape_attributes['name'] = 'polygon'
                shape_attributes['all_points_x'] = approx_contour[i][:, 0][:, 0].tolist()
                # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
                shape_attributes['all_points_y'] = approx_contour[i][:, 0][:, 1].tolist()
                regionsi['shape_attributes'] = shape_attributes
                regionsi['region_attributes'] = region_attributes
                regions[i] = regionsi

            size = os.path.getsize(os.path.join(images_path, elt))
            name = elt + str(size)
            json_elt = {}
            json_elt['filename'] = elt
            json_elt['size'] = str(size)
            json_elt['regions'] = regions
            json_elt['file_attributes'] = {}
            jsonf[name] = json_elt
                
    json.dump(jsonf, js_file) 