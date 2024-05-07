import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 

import tqdm

from nd2reader import ND2Reader
import h5py
import pickle
import json

import datetime

import warnings
warnings.filterwarnings('ignore')

import ffmpeg

from skimage.filters import rank, threshold_otsu, threshold_local#
from skimage import morphology

import scipy 
import skimage

import pandas as pd
import seaborn as sns
import csv

####
##Functions
####

#Incease birghtness if the video is dark (visualization)
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#Metadata parser
def load_metadata(images):
    meta_dict = {}
     # number of locations start 1
    meta_dict["n_fields"] = images.metadata['fields_of_view'].stop

    #number of timeseteps
    meta_dict["n_frames"] = images.metadata['num_frames']

    
    #meta_dict["z_level"] = (np.max(images.metadata['z_coordinates'])-np.min(images.metadata['z_coordinates']))

    meta_dict["z_level"] =  float(images.metadata["z_coordinates"][:images.metadata["z_levels"].stop][-1]-images.metadata["z_coordinates"][:images.metadata["z_levels"].stop][0])/float(images.metadata["z_levels"].stop)
    #number of levels starting from 1
    meta_dict["n_levels"] = images.metadata['z_levels'].stop
    meta_dict["z_step"] = meta_dict["z_level"] /meta_dict["n_levels"]

    #list of channels
    meta_dict["channels"] = images.metadata['channels']

    #number of channels
    meta_dict["n_channels"] = len(meta_dict["channels"])

    meta_dict["m"] = images.metadata['pixel_microns']
    meta_dict["height"] = images.metadata["height"]
    meta_dict["width"] = images.metadata["width"]

    return meta_dict


#Segmentation
def process_frame(img, x_start, y_start):


    frame = img.copy()
    th_num = 500

    frame = scipy.ndimage.gaussian_filter(frame, (5,5))
    frame = frame.astype(int)

    glob_thresh = threshold_otsu(frame)
    binary_local = frame > glob_thresh

    radius = 30   
    footprint = morphology.disk(radius)

    local_otsu = rank.otsu(frame.copy(), footprint)
    lo = frame >= local_otsu    
    l1 = np.zeros_like(lo)
    l1[binary_local] = lo[binary_local]

    # could/should be improved    
    closingCoef = 5
    whiteTopCoef = 5
    dilationCoedf = 5

    l1 = 1-l1
    e = skimage.morphology.closing(l1, skimage.morphology.disk(closingCoef))
    e2 = skimage.morphology.white_tophat(e,skimage.morphology.disk(whiteTopCoef))
    e2 = skimage.morphology.dilation(e2,skimage.morphology.disk(dilationCoedf))

    tmp = e.astype("int")-e2.astype("int")
    #l2 = morphology.erosion(l1,footprint)
    #l3 = morphology.closing(l2,footprint)

    tmp[tmp<0] = 0
    tmp = tmp.astype("uint8")

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

    # draw contours on the original image
    image_copy = np.zeros_like(frame.copy())

    prev = 0
    big_idx = -1

    for idx, i in enumerate(contours):

        current = cv2.contourArea(i)
        
        (x_probe ,y_probe ),radius_probe = cv2.minEnclosingCircle(i)
        
        border = (x_probe < y_start[0]-10) *(y_probe < y_start[1]-10 )*(x_probe > x_start[0]+10)*(y_probe > x_start[1]+10)
        
        area_cond_min =  (current > 1e3) 
        area_cond_max = (current < 2.5e6)

        x_borders =(np.sum(contours[idx][:,0,0] == 0) < th_num )*(np.sum(contours[idx][:,0,0] >= img.shape[0]-1) < th_num)
        y_borders =(np.sum(contours[idx][:,0,1] == 0) < th_num )*(np.sum(contours[idx][:,0,1] >= img.shape[0]-1) < th_num)
        
        if (area_cond_min) & (area_cond_max) & (prev < current) &  (border)  & (x_borders*y_borders):
            
            big_idx = idx
            prev = current


    if big_idx == -1:
        start_pos = x_start
        end_pos = y_start
        area = -1;  x_ = -1; y_ = -1; radius = -1
    else:
        cv2.fillPoly(image_copy, pts = [contours[big_idx]], color=(2**16,0,0))
        (x_,y_),radius = cv2.minEnclosingCircle(contours[big_idx])
        area = cv2.contourArea(contours[big_idx])

        s_1 = int(x_-radius*2.5)
        if s_1<0:
            s_1 = 0

        e_1 = int(x_+radius*2.5)
        if e_1>2304:
            e_1=2304

        s_2 = int(y_+radius*2.5)
        if s_2 >2304:
            s_2 = 2304

        e_2 = int(y_-radius*2.5)
        if e_2 < 0:
            e_2 = 0


        start_pos = (s_1,e_2)
        end_pos = (e_1,s_2)

    #cv2.rectangle(img_sec, start_pos, end_pos, (2**16,0,0), 5)

    #plt.imshow(np.hstack([e, e2, tmp]))
    #plt.show()

    #plt.imshow(np.hstack([img_sec, image_copy]))
    #plt.show()


    return x_, y_, radius, area, start_pos, end_pos, image_copy

#Concatenate dictionaries
def pile_data(current, total_dict, round, color):

    name = "loc_{}_ch_{}".format(round, color)
    total_dict[name] = {}

    total_dict[name]["x"] = []
    total_dict[name]["y"] = []
    total_dict[name]["z"] = []
    total_dict[name]["r"] = []
    total_dict[name]["area"] = []
    total_dict[name]["mask"] = []

    for i in range(len(current)):
        total_dict[name]["x"].append(current[i][0])
        total_dict[name]["y"].append(current[i][1])
        total_dict[name]["r"].append(current[i][2])
        total_dict[name]["area"].append(current[i][3])
        total_dict[name]["z"].append(current[i][4])
        total_dict[name]["mask"].append(current[5])

    return total_dict

#Draw rectangle to images
def mousePoints(event,x,y,flags,param):
    #Crop image
    global refPt
    global img
    global final_boundaries
    global stopper
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        final_boundaries.append((refPt[0],refPt[1]))
        stopper = True
        cv2.imshow("win", img)
        print("two clicks!")
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        clone = img.copy()
        cv2.rectangle(clone, refPt[0], (x, y), (0, 255, 0), 4)
        cv2.imshow("win", clone)
