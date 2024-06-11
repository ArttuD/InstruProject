import os
import numpy as np
import cv2 
import glob

import tqdm

from nd2reader import ND2Reader
import pickle
import json

import scipy 
import skimage

from skimage.morphology import closing, dilation
from skimage.morphology import disk

from tools.func import *

def process_FL(img_bf, img_fl, x_start, y_start):

    img_fl = scipy.ndimage.gaussian_filter(img_fl, (3,3))
    tuned_fl = Kittler_16(img_fl, np.empty_like(img_fl))
    tuned_fl = closing(tuned_fl,disk(5))
    frame = (tuned_fl/(2**16)*2**8).astype("uint8")


    contours, hierarchy = cv2.findContours(image=frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

    prev = 0
    idx_big = -1

    for nmr, i in enumerate(contours):
        if check_contour(i, prev,  x_start, y_start, img_bf.shape[0]):
            prev = cv2.contourArea(i)
            idx_big = nmr

    #out_vis = cv2.addWeighted(img_bf,1.0,np.stack((frame,frame,frame), axis = -1)*255,0.25,5)
    img_bf = (img_bf/(2**16)*2**8).astype("uint8")
    img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)

    out_vis = img_bf
    if idx_big == -1:
        contour = []
        (x, y) = (-1,-1)
        r = -1
    else:
        cv2.drawContours(out_vis, contours, idx_big, (0, 0, 255), 3)
        (x, y), r = cv2.minEnclosingCircle(contours[idx_big])
        x_start, y_start = check_box(x, y, r)

    
    return out_vis, x, y, r, prev, idx_big, contours, x_start, y_start

def process_BF(img_bf, x_start, y_start):

    tuned_bf = scipy.ndimage.gaussian_filter(img_bf.copy(), (3,3))
    th = yen_filter_16(tuned_bf)
    tuned_bf = tuned_bf > th
    tuned_bf = 1-tuned_bf
    
    tuned_bf = dilation(tuned_bf, disk(10))
    tuned_bf = closing(tuned_bf, disk(10))
    #tuned_bf = closing(tuned_bf, disk(3))
    frame = tuned_bf.astype("uint8") #(tuned_bf/(2**16)*2**8).astype("uint8")
    #plt.imshow(frame)
    #plt.show()

    contours, hierarchy = cv2.findContours(image=frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

    prev = 0
    idx_big = -1

    for nmr, i in enumerate(contours):
        if check_contour(i, prev,  x_start, y_start, img_bf.shape[0]):
            prev = cv2.contourArea(i)
            idx_big = nmr

    img_bf = (img_bf/(2**16)*2**8).astype("uint8")
    img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)
    out_vis = img_bf

    if idx_big == -1:
        contour = []
        (x, y) = (-1,-1)
        r = -1
    else:
        cv2.drawContours(out_vis, contours, idx_big, (0, 0, 255), 3)
        (x, y), r = cv2.minEnclosingCircle(contours[idx_big])
        x_start, y_start = check_box(x, y, r)

    return out_vis, x, y, r, prev, idx_big, contours, x_start, y_start

root_path = "F:/instru_projects/TimeLapses/u-wells/*"
target_paths = glob.glob(os.path.join(root_path, "*.nd2"))

root_path_2 = "E:/instru_projects/TimeLapses/u-wells/*"
target_paths = target_paths + glob.glob(os.path.join(root_path_2, "*.nd2"))

target_paths_FL = glob.glob(os.path.join(root_path, "*mCherry.nd2"))
target_paths_FL = target_paths_FL + glob.glob(os.path.join(root_path_2, "*mCherry.nd2"))

ignore_paths = []

target_paths = [
    "F:/instru_projects/TimeLapses/u-wells/collagen/240304_timelapses_collagen_3lines_48h_spheroidseeded.nd2",
    "F:/instru_projects/TimeLapses/u-wells/collagen/240226_timelapses_collagen_3lines_72h_spheroidseeded.nd2",
    "F:/instru_projects/TimeLapses/u-wells/collagen/240306_timelapses_collagen_3lines_115h_spheroidseeded.nd2",
    "F:/instru_projects/TimeLapses/u-wells/IPN/230417_timelapses_IPN3mM_3lines_63h_culture.nd2",
    "F:/instru_projects/TimeLapses/u-wells/IPN/230418_timelapses_IPN3mM_3lines_91h_culture.nd2"
]

with open('./dataStore/metalib.json', 'r') as f:
  own_meta = json.load(f)

scaler = 350

for video_path in tqdm.tqdm(target_paths, total=len(target_paths)):

    print("Analyzing: ", video_path)
    video_name = os.path.split(video_path)[-1][:-4]
    root_path = os.path.split(video_path)[0]
    results = os.path.join(root_path, "results_{}".format(video_name))
    os.makedirs(results, exist_ok=True)
    parts = os.path.split(video_path)[-1].split("_")
    day = str(parts[0])
    
    if day not in own_meta.keys():
        print(day, "Not in keys, skipping")
        continue

    coords = own_meta[day]["coords"]
    track_list = []
    total_dict = {}

    with ND2Reader(video_path) as images:

        metas = load_metadata(images)
        if metas["n_channels"] == 2:
            FL_flag = True
        else:
            FL_flag = False

        if FL_flag:
            for d in range(len(metas["channels"])):
                if metas["channels"][d] == 'BF':
                    idx_bf = d
                elif metas["channels"][d] == 'Red':
                    idx_fl = d

        for k in range(metas["n_fields"]): #
            
            if k < len( own_meta["240304"]["cell"]):
                line_name = own_meta["240304"]["cell"][k]
            else:
                line_name = "unknown"
            
            if (day == "230418") & (k == 2):
                pass 

            out_name = os.path.join(results,'{}_{}_{}.mp4'.format(os.path.split(video_path)[1][:-4], (k), (line_name) ) )
            out_process = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), 5, (2304,2304))

            x_final = coords[k][0] #(0,2304)
            y_final = coords[k][1] #(2304,0)
            
            #try:

            for j in range(metas["n_frames"]):

                idx = 0
                prev = 0

                if FL_flag:

                    for z in range(metas["n_levels"]):
                        current = images.get_frame_2D(c=1, t=j, z=z, x=0, y=0, v=k)
                        current = current[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                        current = skimage.measure.blur_effect(current)

                        if current > prev:
                            idx = z

                    img_fl = images.get_frame_2D(c=idx_fl, t=j, z=idx, x=0, y=0, v=k)
                    img_bf = images.get_frame_2D(c=idx_bf, t=j, z=idx, x=0, y=0, v=k)

                    out_vis, x, y, r, prev, big_idx, contours, x_final, y_final = process_FL(img_bf, img_fl, x_final, y_final)
                else:

                    for z in range(metas["n_levels"]):

                        current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)
                        current = current[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                        current = cv2.Laplacian(current, cv2.CV_64F).var()
                        
                        if current > prev:
                            prev = current
                            idx = z

                    img_bf = images.get_frame_2D(c=0, t=j, z=idx, x=0, y=0, v=k)
                    out_vis, x, y, r, prev, big_idx, contours, x_final, y_final = process_BF(img_bf, x_final, y_final)

                out_process.write(out_vis)
            #except:
            #    out_process.release()
            #    pass

            track_list.append([x*metas["m"], y*metas["m"], r*metas["m"], prev*metas["m"]**2, (idx)*metas["z_step"], contours])
            try:
                total_dict = pile_data(track_list, total_dict, k, 1)
            except:
                continue
            
            with open(os.path.join(results,'{}_detections.pkl'.format(os.path.split(video_path)[1][:-4])), 'wb') as f:
                pickle.dump(total_dict, f)

            out_process.release()
            
           
        