import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import glob
import tqdm
from nd2reader import ND2Reader
import json
import skimage
from tools.func import *



class Change_Level():
    
    def __init__(self, fig, img_ax, img_plots, out_path, meta):

        self.figure = fig
        self.meta = meta
        self.img_plots = img_plots

        if isinstance(img_ax,np.ndarray):
            self.img_ax = img_ax.ravel()
        else:
            self.img_ax = [img_ax]

        self.changed = False
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self)
        self.close = self.figure.canvas.mpl_connect('close_event', self.on_close)

        self.out_path = out_path
        self.manual = [0]*self.meta["n_frames"]
        self.names = ['','Manual']


    @staticmethod
    def find(axes,ax):

        for idx,i in enumerate(axes):
            if i == ax:
                return idx
        return 0

    def on_close(self,event):
        print("donee")
        #self.focus_lines[0].figure.savefig('{}'.format(os.path.join(self.out_path,'radius_estimate.png')))

    def get_rads(self):
        return self.manual

    def __call__(self, event):
        x = event.xdata
        y = event.ydata
        if event.inaxes in self.img_ax:
            a_id = self.find(self.img_ax, event.inaxes)
            self.manual[a_id] +=1
            self.img_ax[a_id].set_title(self.names[self.manual[a_id]%2])
            self.img_ax[a_id].figure.canvas.draw()

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.cid)
        self.figure.canvas.mpl_disconnect(self.close)

class focus_detector():

    def __init__(self) -> None:
        

        self.target_paths, self.target_paths_FL = self.find_paths()

        with open('./dataStore/metalib.json', 'r') as f:
            self.own_meta = json.load(f)   

        self.focus_dict = {}

        self.alpha = 0.3
        self.beta = ( 1.0 - self.alpha )


    def find_paths(self):

        root_path = "D:/instru_projects/TimeLapses/u-wells/*"
        target_paths = glob.glob(os.path.join(root_path, "*.nd2"))

        #root_path_2 = "F:/instru_projects/TimeLapses/u-wells/*"
        #target_paths = target_paths + glob.glob(os.path.join(root_path_2, "*.nd2"))

        target_paths_FL = glob.glob(os.path.join(root_path, "*mCherry.nd2"))

        #target_paths_FL = target_paths_FL + glob.glob(os.path.join(root_path_2, "*.nd2"))

        for i in target_paths:
            print(i)

        return target_paths, target_paths_FL


    def process_pipe(self):
                
        for video_path in tqdm.tqdm(self.target_paths, total=len(self.target_paths)):
            self.focus_dict = {}
            print("Analyzing:", video_path)
            video_name = os.path.split(video_path)[-1][:-4]
            root_path = os.path.split(video_path)[0]
            results = os.path.join(root_path, "results_{}".format(video_name))
            os.makedirs(results, exist_ok=True)

            parts = os.path.split(video_path)[-1].split("_")
            day = str(parts[0])
            self.coords = self.own_meta[day]["coords"]   

            if day not in self.own_meta.keys():
                print(day, "Not in keys, skipping")
                continue

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

                for k in range(metas["n_fields"]): 
                    
                    if (day == "230418") & (k == 2):
                        pass 

                    x_final = self.coords[k][0] #(0,2304)
                    y_final = self.coords[k][1] #(2304,0)

                    max_indices = []
                    all_focus = []
                    img_plots = [None]*metas["n_frames"]
                    sub = 0

                    plot_ind = find_plot_size(metas["n_frames"])
                    fig, ax = plt.subplots(plot_ind,plot_ind,figsize=(plot_ind,plot_ind))

                    for j in range(metas["n_frames"]):


                        dets = np.zeros(metas["n_levels"])

                        if FL_flag:
                            for z in range(metas["n_levels"]):
                                try:
                                    current = images.get_frame_2D(c=idx_fl, t=j, z=z, x=0, y=0, v=k)
                                except:
                                    j -= 1
                                    current = images.get_frame_2D(c=idx_fl, t=j, z=z, x=0, y=0, v=k)

                                current = current[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                                current = skimage.measure.blur_effect(current)
                                dets[z] = current

                            dets[np.isnan(dets)] = -np.inf
                            idx = np.argmax(dets)

                            img_fl = images.get_frame_2D(c=idx_fl, t=j, z=idx, x=0, y=0, v=k)
                            img_bf = images.get_frame_2D(c=idx_bf, t=j, z=idx, x=0, y=0, v=k)
                            im = cv2.addWeighted( img_bf, self.alpha, img_fl, self.beta, 0.0, 0.0)
                            im = (im/(2**16)*2**8).astype("uint8")
                            img_vis = np.stack((im,im,im), axis = -1)

                        else:
                            for z in range(metas["n_levels"]):
                                try:
                                    current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)
                                except:
                                    j -= 1
                                    current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)

                                current = current[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                                current = cv2.Laplacian(current, cv2.CV_64F).var()

                                dets[z] = current

                            dets[np.isnan(dets)] = -np.inf
                            idx = np.argmax(dets)

                            img_bf = images.get_frame_2D(c=0, t=j, z=idx, x=0, y=0, v=k)

                            img_bf = (img_bf/(2**16)*2**8).astype("uint8")
                            img_vis = np.stack((img_bf, img_bf, img_bf), axis = -1)

                        if j !=0 and j%plot_ind == 0:
                            sub += 1

                        img_cropped_vis = img_vis[x_final[1]:y_final[1], x_final[0]:y_final[0]]
                        #img_cropped_vis = skimage.exposure.equalize_hist(img_cropped_vis)

                        img_plots[j] = ax[sub,j%plot_ind].imshow(cv2.resize(img_cropped_vis, (512,512)))


                        #focus_lines.append(l1)
                        all_focus.append(dets)
                        max_indices.append(idx)

                                                    
                    self.handler =  Change_Level(fig, ax, img_plots, "./dataStore", metas)
                    fig.tight_layout()
                    plt.show()
                    
                    self.handler.disconnect()
                    response_vals = np.array(self.handler.manual)

                    #self.handler.disconnect()

                    stop_measuring = False

                    for id_repair in np.arange(metas["n_frames"]):
                            
                        if response_vals[id_repair] == 0:
                            continue

                        
                        start_idx = max_indices[id_repair]
                        choosing = True
                        
                        while choosing:
                            if FL_flag:
                                img_bf = images.get_frame_2D(c=idx_bf, t=id_repair, z=start_idx, x=0, y=0, v=k)
                            else:
                                img_bf = images.get_frame_2D(c=0, t=id_repair, z=start_idx, x=0, y=0, v=k)

                            img_bf = (img_bf/(2**16)*2**8).astype("uint8")
                            img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)

                            windowText = "t={}, z={}, v={}".format( id_repair, start_idx, k)

                            #img_bf = skimage.exposure.equalize_hist(img_bf)

                            cv2.imshow(windowText, cv2.resize(img_bf, (520,520)))
                            # add wait key. window waits until user presses a key
                            kk = cv2.waitKey(0)
                            # and finally destroy/close all open windows
                            if kk == 119: #Move up 
                                start_idx += 1
                                if start_idx == metas["n_levels"]:
                                    start_idx -=1
                            elif kk == 115: #Move down
                                start_idx -= 1
                                if start_idx == -1:
                                    start_idx +=1
                            elif kk == 113: #Exit timewwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwstamp
                                choosing = False
                            elif kk == 101: #Exit timestamp and move to next location
                                choosing = False
                                stop_measuring = True
                                start_idx = -1
                            else:
                                print("incorrect key", k)
                        
                            cv2.destroyAllWindows()

                        max_indices[id_repair] = start_idx

                        if stop_measuring:
                            stop_measuring = False
                            break

                        
                    self.focus_dict[k] = max_indices
                    #print(self.focus_dict)

                    with open(os.path.join(results, 'focus_indixes.pkl'), 'wb') as f:
                        pickle.dump(self.focus_dict, f)

                
        return 1


if __name__ == "__main__":

        focuser = focus_detector()
        success = focuser.process_pipe()
        print("Done")
        exit()