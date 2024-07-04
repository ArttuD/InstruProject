

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
        pass
        #print("donee")
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


class PostProcess():



    def __init__(self) -> None:
        

        self.target_paths = self.find_paths()

        with open('./dataStore/metalib.json', 'r') as f:
            self.own_meta = json.load(f)   

        self.scaled_size = 1020

        self.scaler = 2304/self.scaled_size

        self.data_dict = None
        self.current_key = None


    def find_paths(self):

        root_path = "D:/instru_projects/TimeLapses/u-wells/*"
        target_paths = glob.glob(os.path.join(root_path, "*.nd2"))

        root_path_2 = "E:/instru_projects/TimeLapses/u-wells/*"
        target_paths += glob.glob(os.path.join(root_path_2, "*.nd2"))

        root_path_2 = "F:/instru_projects/TimeLapses/u-wells/*"
        target_paths += glob.glob(os.path.join(root_path_2, "*.nd2"))

        root_path_2 = "G:/instru_projects/TimeLapses/u-wells/*"
        target_paths += glob.glob(os.path.join(root_path_2, "*.nd2"))

        for i in target_paths:
            print(i)

        return target_paths
    
    def click_event(self, event, x, y, flags, params):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            #print(f'({x},{y})')

            self.pts.append([int(x*self.scaler),int(y*self.scaler)])
            pts_ = np.array(self.pts).reshape((-1, 1, 2))

            self.img_ = self.img_bf.copy()
            self.img_ = cv2.polylines(self.img_, [pts_], 
                            True, (255, 0, 0), 2)
            
            cv2.imshow("window", cv2.resize(self.img_, (self.scaled_size,self.scaled_size)) )

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.pts) > 0:
                del self.pts[-1]
                pts_ = np.array(self.pts).reshape((-1, 1, 2))

                self.img_ = self.img_bf.copy()
                self.img_ = cv2.polylines(self.img_, [pts_], 
                                True, (255, 0, 0), 2)
                
                cv2.imshow("window", cv2.resize(self.img_, (self.scaled_size,self.scaled_size)) )

    def pipe(self):

        for video_path in tqdm.tqdm(self.target_paths, total=len(self.target_paths)):

            video_name = os.path.split(video_path)[-1][:-4]
            root_path = os.path.split(video_path)[0]
            results = os.path.join(root_path, "results_{}".format(video_name))

            parts = os.path.split(video_path)[-1].split("_")
            day = str(parts[0])
            self.coords = self.own_meta[day]["coords"]

            focus_path = glob.glob(os.path.join(results, "focus_indixes.pkl")) #*_indixes.pkl
            with open(focus_path[0], 'rb') as f:
                self.focus_dict = pickle.load(f)   

            pickel_path = os.path.join(results,"{}_detections.pkl".format(video_name))
            with open(pickel_path, 'rb') as f:
                self.data_dict = pickle.load(f)

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
                else:
                    idx_bf = 0



                for k in range(metas["n_fields"]): 

                    self.current_key = "loc_{}_ch_{}".format(k, 1)
                    if (day == "230418") & (k == 2):
                        pass 

                    sub = 0
                    j_ = 0
                    img_plots = [None]*9 #*metas["n_frames"]
                    self.pts = []
                    
                    plot_ind = find_plot_size(metas["n_frames"])

                    if self.current_key in self.data_dict.keys():
                        masks = self.data_dict[self.current_key]['mask']
                        idx_larges = self.data_dict[self.current_key]["big_idx"]
                    elif "loc_{}_ch_{}".format(k, 0) in self.data_dict.keys():
                        self.current_key = "loc_{}_ch_{}".format(k, 0)  
                        masks = self.data_dict[self.current_key]['mask']
                        idx_larges = self.data_dict[self.current_key]["big_idx"]
                    else:
                        continue


                    fig, ax = plt.subplots(3,3,figsize=(plot_ind,plot_ind))

                    for j in range(metas["n_frames"]):

                        idx = int(self.focus_dict[k][j])

                        if (idx == -1):
                            self.handler =  Change_Level(fig, ax, img_plots, "./dataStore", metas)
                            fig.tight_layout()
                            plt.show()
                            
                            self.handler.disconnect()
                            self.response_vals = np.array(self.handler.manual)

                            self.process(metas, idx_bf, k, j-8, images)

                            break


                        mask = masks[j][idx_larges[j]]
                        

                        img_bf = images.get_frame_2D(c=idx_bf, t=j, z=idx, x=0, y=0, v=k)
                        img_bf = (img_bf/(2**16)*2**8).astype("uint8")
                        img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)
                        
                        if mask.shape[0] > 0:
                            cv2.drawContours(img_bf, [mask], 0, (0, 0, 255), 3)

                        if j_ !=0 and j_%3 == 0:
                            sub += 1

                        
                        img_plots[j_] = ax[sub,j_%3].imshow(cv2.resize(img_bf, (512,512)))


                        if (sub == 2 ) & (j_ == 8):
         
                            self.handler =  Change_Level(fig, ax, img_plots, "./dataStore", metas)
                            fig.tight_layout()
                            fig.suptitle("Frames {}/{}".format(j+1, metas["n_frames"]))
                            plt.show()
                            
                            self.handler.disconnect()
                            self.response_vals = np.array(self.handler.manual)

                            self.process(metas, idx_bf, k, j-8, images)

                            sub = 0
                            j_ = 0
                            img_plots = [None]*9#*metas["n_frames"]
                            self.pts = []
                            fig, ax = plt.subplots(3,3,figsize=(plot_ind,plot_ind))
                        else:
                            j_ += 1

                        if (j == metas["n_frames"]-1):
                            self.handler =  Change_Level(fig, ax, img_plots, "./dataStore", metas)
                            fig.suptitle("Frames {}/{}".format(j+1, metas["n_frames"]))
                            fig.tight_layout()
                            plt.show()
                            
                            self.handler.disconnect()
                            self.response_vals = np.array(self.handler.manual)

                            self.process(metas, idx_bf, k, j-8, images)

                            break

                    with open(os.path.join(results,'{}_corrected_detections.pkl'.format(os.path.split(video_path)[1][:-4])), 'wb') as f:
                        pickle.dump(self.data_dict, f)

                    with open(os.path.join(results,'corrected_focus_indixes.pkl'), 'wb') as f:
                        pickle.dump(self.focus_dict, f)

    def process(self, metas, idx_bf, loc, idx_frame, images):

        # create a window
        cv2.namedWindow('window')
        # bind the callback function to window
        cv2.setMouseCallback('window', self.click_event)

        for id_repair in np.arange(metas["n_frames"]):
                
            if self.response_vals[id_repair] == 0:
                continue
            
            id_repair += idx_frame

            self.pts = []
            choosing = True
            start_idx = int(self.focus_dict[loc][id_repair])

            #self.changed_countour = False

            while choosing:

                img_bf = images.get_frame_2D(c=idx_bf, t=id_repair, z=start_idx, x=0, y=0, v=loc)

                img_bf = (img_bf/(2**16)*2**8).astype("uint8")
                img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)

                self.img_bf = img_bf.copy() 
                self.img_ = img_bf.copy()

                windowText = "t={}, z={}, v={}".format(id_repair, start_idx, loc)

                # put coordinates as text on the image
                cv2.putText(self.img_, windowText,(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
                cv2.imshow("window", cv2.resize(self.img_, (self.scaled_size,self.scaled_size)) )

                # add wait key. window waits until user presses a key
                kk = cv2.waitKey(0)
                # and finally destroy/close all open windows
                if kk == 113: #Exit 
                    choosing = False
                elif kk == 101: #clear
                    self.pts = []
                elif kk == 119: #Move up 
                    start_idx += 1
                    if start_idx == metas["n_levels"]:
                        start_idx -=1
                elif kk == 115: #Move down
                    start_idx -= 1
                    if start_idx == -1:
                        start_idx +=1
                else:
                    print("incorrect key", kk)

            #kk = cv2.waitKey(0)
            if len(self.pts) > 0:
                pts_ = np.array(self.pts).reshape((-1, 1, 2))
                self.img_ = np.zeros_like(self.img_bf)
                self.img_ = cv2.polylines(self.img_, [pts_], True, (255, 0, 0), 2)
                imgB = cv2.cvtColor(self.img_.copy(), cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(image=imgB, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                c = max(contours, key=cv2.contourArea)
                self.data_dict[self.current_key]['mask'][idx_frame] = [c]
                self.data_dict[self.current_key]['big_idx'][idx_frame] = 0

            if int(self.focus_dict[loc][id_repair]) != start_idx:
                self.focus_dict[loc][id_repair] = start_idx

            self.changed_countour = False
            
        cv2.destroyAllWindows()



if __name__ == "__main__":

        process = PostProcess()
        success = process.pipe()
        print("Done")
        exit()