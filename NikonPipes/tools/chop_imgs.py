

from glob import glob
import os
import cv2
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
from tools.func import *
import tifffile as tiff


root_path = "E:/instru_projects/TimeLapses/u-wells/collagen"
target_paths = glob(os.path.join(root_path, "*.nd2"))



for video_path in target_paths:

    video_name = os.path.split(video_path)[-1][:-4]
    root_path = os.path.split(video_path)[0]
    results = os.path.join(root_path, "results_{}".format(video_name))
    os.makedirs(results, exist_ok=True)
    parts = os.path.split(video_path)[-1].split("_")
    day = str(parts[0])

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


        for k in range(metas["n_fields"]): #range(metas["n_fields"])metas["n_fields"]-7,
            k = np.abs(k)
            xs = list(range(20))
            xs[0::5]
            for j in xs: ##metas["n_frames"]

                if FL_flag:
                    for z in [10]: #tqdm.tqdm(range(metas["n_levels"]), total = metas["n_levels"])
                        try:
                            img_bf = images.get_frame_2D(c=idx_bf, t=j, z=z, x=0, y=0, v=k)
                        except:
                            j-=1
                            img_bf = images.get_frame_2D(c=idx_bf, t=j, z=z, x=0, y=0, v=k)
                        try:
                            img_fl = images.get_frame_2D(c=idx_fl, t=j, z=z, x=0, y=0, v=k)
                        except:
                            j-=1
                            img_fl = images.get_frame_2D(c=idx_fl, t=j, z=z, x=0, y=0, v=k)
                        
                        tiff.imsave("./images/bf/bf_image_{}_{}_{}_{}.tif".format(day, k, j, z), img_bf) 
                        tiff.imsave("./images/fl/fl_image_{}_{}_{}_{}.tif".format(day, k, j, z), img_fl) 
                else:
                    for z in [0,5,10,15,20]: #tqdm.tqdm(range(metas["n_levels"]), total = metas["n_levels"]):
                        try:
                            img_bf = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)
                        except:
                            j-=1
                            img_bf = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)
                        
                        tiff.imsave("./images/bf/bf_image_{}_{}_{}_{}.tif".format(day, k, j, z), img_bf) 

                

