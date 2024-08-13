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
import argparse


def process_FL(img_bf, img_fl, x_start, y_start, otsu_flag):


    img_fl = scipy.ndimage.gaussian_filter(img_fl, (3,3))

    if otsu_flag:
        img_ = skimage.exposure.equalize_adapthist(img_fl, clip_limit=0.03)
        img_ = cv2.normalize(img_, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        ret2,th2 = cv2.threshold(img_.astype("uint8"),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th2 = dilation(th2, disk(5))
        frame = closing(th2, disk(5))

        #frame = dilation(frame, disk(5))
        #frame = closing(frame, disk(5))
    else:
        tuned_fl = Kittler_16(img_fl, np.empty_like(img_fl))
        #tuned_fl = closing(tuned_fl,disk(5))
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

def process_BF(img_bf, x_start, y_start, local_flag):



    if local_flag:
        tuned_bf = local_th(img_bf.copy())
    else: 
        tuned_bf = scipy.ndimage.gaussian_filter(img_bf.copy(), (3,3))
        th = yen_filter_16(tuned_bf)
        tuned_bf = tuned_bf > th
        tuned_bf = 1-tuned_bf
    
    tuned_bf = dilation(tuned_bf, disk(5)) #10
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

    #cv2.rectangle(out_vis, x_start, y_start, (255,0,0), 3)
    #cv2.imshow('window' ,cv2.resize(out_vis, (520,520)))
    #cv2.waitKey(0)

    #cv2.destroyAllWindows()

    return out_vis, x, y, r, prev, idx_big, contours, x_start, y_start


parser = argparse.ArgumentParser(
    description="""Download results in the folder and ouputs results
                """)
parser.add_argument('--path','-p',required=False,default= None, help='Path to folder. eg. C:/data/imgs')
parser.add_argument('--blur','-b',required=False,default= [[]], help='blurred images')

#Save arguments
args = parser.parse_known_args()[0]


otsu_list = args.blur
otsu_list_counter = 0
otsu_flag = False

if args.path:
    target_paths = [args.path]
else:
    target_paths = glob.glob("D:/instru_projects/TimeLapses/u-wells/*/*.nd2") 
    target_paths += glob.glob("F:/instru_projects/TimeLapses/u-wells/*/*.nd2")
    target_paths += glob.glob("G:/instru_projects/TimeLapses/u-wells/*/*.nd2") 
    target_paths += glob.glob("E:/instru_projects/TimeLapses/u-wells/*/*.nd2")
    target_paths += glob.glob("H:/instru_projects/TimeLapses/u-wells/*/*.nd2")

ignore_paths = []
for i in range(len(target_paths)):
    print(target_paths[i])


local_flag = False 

with open('./dataStore/metalib.json', 'r') as f:
  own_meta = json.load(f)

scaler = 350

for video_path in tqdm.tqdm(target_paths, total=len(target_paths)):

    otsu_files = otsu_list[otsu_list_counter]
    otsu_list_counter += 1

    print("Analyzing: ", video_path)
    video_name = os.path.split(video_path)[-1][:-4]
    root_path = os.path.split(video_path)[0]
    results = os.path.join(root_path, "results_{}".format(video_name))
    os.makedirs(results, exist_ok=True)
    parts = os.path.split(video_path)[-1].split("_")
    day = str(parts[0])

    if day == "240304":
        local_flag = True

    focus_path = glob.glob(os.path.join(results, "*_indixes.pkl"))

    if len(focus_path) == 0:
        focus_flag = False
    else:
        focus_flag = True
        with open(focus_path[0], 'rb') as f:
            focus_dict = pickle.load(f)

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
            
            flag_tracking = True

            if k in otsu_files:
                otsu_flag = True
            else:
                otsu_flag = False

            if k < len(own_meta[day]["cell"]):
                line_name = own_meta[day]["cell"][k]
            else:
                line_name = "unknown"
            
            #if (day == "230418") & (k == 1):
            #    pass 

            out_name = os.path.join(results,'{}_{}_{}.mp4'.format(os.path.split(video_path)[1][:-4], (k), (line_name) ) )
            out_process = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), 5, (2304,2304))

            x_final = coords[k][0] #(0,2304)
            y_final = coords[k][1] #(2304,0)
            
            #try:

            for j in range(metas["n_frames"]):
                init_timestep = j
                idx = 0
                prev = 0

                if FL_flag:
                    if focus_flag:
                        idx = int(focus_dict[k][j])
                        if idx == -1: #Drifts out of focus, stop tracking
                            flag_tracking = False
                            break
                    else:
                        for z in range(metas["n_levels"]):
                            try:
                                current = images.get_frame_2D(c=idx_fl, t=j, z=z, x=0, y=0, v=k)
                            except:
                                j-=1
                                current = images.get_frame_2D(c=idx_fl, t=j, z=z, x=0, y=0, v=k)

                            current = current[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                            current = skimage.measure.blur_effect(current)

                            if current > prev:
                                idx = z

                    img_fl = images.get_frame_2D(c=idx_fl, t=j, z=idx, x=0, y=0, v=k)
                    img_bf = images.get_frame_2D(c=idx_bf, t=j, z=idx, x=0, y=0, v=k)

                    out_vis, x, y, r, prev, big_idx, contours, x_final, y_final = process_FL(img_bf, img_fl, x_final, y_final, otsu_flag)
                else:
                    if focus_flag:
                        idx = int(focus_dict[k][j])
                        if idx == -1: #Drifts out of focus, stop tracking
                            flag_tracking = False
                            break
                    else:
                        for z in range(metas["n_levels"]):

                            try:
                                current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)
                            except:
                                #print("Cannot get frame", j)
                                j-=1
                                current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)

                            current = current[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                            current = cv2.Laplacian(current, cv2.CV_64F).var()
                            
                            if current > prev:
                                prev = current
                                idx = z

                    img_bf = images.get_frame_2D(c=0, t=j, z=idx, x=0, y=0, v=k)
                    out_vis, x, y, r, prev, big_idx, contours, x_final, y_final = process_BF(img_bf, x_final, y_final, local_flag)

                out_process.write(out_vis)

                if big_idx == -1:
                    try:
                        new = track_list[-1]
                        new[-1] += 1
                        track_list.append(new)
                    except:
                        print("First frame failed")
                        track_list.append([x*metas["m"], y*metas["m"], r*metas["m"], prev*metas["m"]**2, (idx)*metas["z_step"], contours, big_idx, init_timestep])
                else:
                    track_list.append([x*metas["m"], y*metas["m"], r*metas["m"], prev*metas["m"]**2, (idx)*metas["z_step"], contours, big_idx, init_timestep])

            total_dict = pile_data(track_list, total_dict, k, 1)
            track_list = []
            local_flag = False
            
            with open(os.path.join(results,'{}_detections.pkl'.format(os.path.split(video_path)[1][:-4])), 'wb') as f:
                pickle.dump(total_dict, f)

            out_process.release()
            
           
        