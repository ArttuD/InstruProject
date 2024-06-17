import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 
import glob

import tqdm

from nd2reader import ND2Reader
import json


import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from tools.func import *

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



skip_existing = False
map_coord = True
parse_flag = True

root_path = "F:/instru_projects/TimeLapses/u-wells/*"
target_paths = glob.glob(os.path.join(root_path, "*.nd2"))

root_path_2 = "E:/instru_projects/TimeLapses/u-wells/*"
target_paths = target_paths + glob.glob(os.path.join(root_path_2, "*.nd2"))

target_paths_FL = glob.glob(os.path.join(root_path, "*mCherry.nd2"))
target_paths_FL = target_paths_FL + glob.glob(os.path.join(root_path_2, "*mCherry.nd2"))


with open('C:/Users/lehtona6/codes/InstruProject/NikonPipes/dataStore/metalib.json', 'r') as f:
  own_meta = json.load(f)


coord_dict = {}

for video_path in target_paths:

    parts = os.path.split(video_path)[-1].split("_")
    day = str(parts[0])
    matrix = parts[2]
    n_ines = parts[3][0]
    start_time = int(parts[4][:-1])
    comments = parts[5][:-3]

    if day not in own_meta.keys():
        own_meta[day] = {}
    elif skip_existing:
        continue

    own_meta[day]["matrix"]  = matrix
    own_meta[day]["n_cells"]  = n_ines
    own_meta[day]["dim"] = "3D"
    own_meta[day]["incubation_time"] = start_time
    own_meta[day]["other"] = "other"

    own_meta = parse_raw_dict(day, video_path, own_meta)
    
    if map_coord:

        with ND2Reader(video_path) as images:

            coords = []
            final_boundaries = []
            metas = load_metadata(images)
            z_levels = metas["n_levels"]
            vis_level = int(z_levels/2)
            stopper = False

            for i in range(metas["n_fields"]):

                cv2.namedWindow('win',cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("win", mousePoints)

                img = images.get_frame_2D(c=0, t=0, z=vis_level, x=0, y=0, v=i)
                cv2.imshow("win",img)

                k = cv2.waitKey(0)

                if k == ord("q"):  # Press q to quit
                    break

                if stopper == True:

                    stopper = False
                    coords.append(final_boundaries)
                    final_boundaries = []
                    cv2.destroyAllWindows()


            cv2.destroyAllWindows()

            if k == ord("q"):  # Press q to quit
                break

        final_coords = []

        for i in range(len(coords)):
            final_coords.append([[coords[i][0][0][0], coords[i][0][0][1]], [coords[i][0][1][0], coords[i][0][1][1]]])

        
        own_meta[day]["coords"] = final_coords
        
        if k == ord("q"):  # Press q to quit
            break


    with open('./dataStore/metalib.json', 'w', encoding='utf-8') as f:
        json.dump(own_meta, f, ensure_ascii=False, indent=4)


with open('C:/Users/lehtona6/codes/InstruProject/NikonPipes/dataStore/metalib.json', 'r') as f:
    own_meta = json.load(f)

for video_path in target_paths:

    parts = os.path.split(video_path)[-1].split("_")
    day = str(parts[0])
    own_meta = parse_raw_dict(day, video_path, own_meta)

    with ND2Reader(video_path) as images:
        metas = load_metadata(images)

    for k in metas.keys():
        own_meta[day][k] = metas[k]

with open('./dataStore/metalib.json', 'w', encoding='utf-8') as f:
    json.dump(own_meta, f, ensure_ascii=False, indent=4)
