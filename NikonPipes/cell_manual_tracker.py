import glob
import os
import tqdm
import json
from tools.func import *
from tools.saver import Cells
from nd2reader import ND2Reader


class manual_tracker():

    def __init__(self) -> None:


        self.target_paths = self.find_path()
        self.scaled_size = 1024
        self.converter = 2304/self.scaled_size


        with open('./dataStore/metalib.json', 'r') as f:
            self.own_meta = json.load(f)   

    def find_path(self):
        
        root_path = "D:/instru_projects/TimeLapses/u-wells/*"
        target_paths = glob.glob(os.path.join(root_path, "*.nd2"))

        root_path_2 = "F:/instru_projects/TimeLapses/u-wells/*"
        target_paths = target_paths + glob.glob(os.path.join(root_path_2, "*.nd2"))

        for i in target_paths:
            print(i)

        return target_paths
    
    def check_logged(self):
        log_files = glob.glob(os.path.join(self.results, "log_single.npy"))
        if len(log_files) == 0:
            return -1
        else:
            log_file = np.load("log_single.npy")
            self.n_start = log_file[0]; self.t_start = log_file[1]
            return 1
        
    def save_logged(self):
        info = np.array((self.k, self.t_start))
        np.save( os.path.join(self.results,"log_single.npy"), info)
        
    def fetch_image(self, images, j, z, k):
        
        if self.FL_flag:
            try:
                current = images.get_frame_2D(c=self.idx_bf, t=j, z=z, x=0, y=0, v=k)
            except:
                j -= 1
                if j == -1:
                    j = 0
                    z += 1

                current = images.get_frame_2D(c=self.idx_bf, t=j, z=z, x=0, y=0, v=k)
        else:
            try:
                current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)
            except:
                j -= 1
                if j == -1:
                    j = 0
                    z += 1
                current = images.get_frame_2D(c=0, t=j, z=z, x=0, y=0, v=k)

        img_bf = (current/(2**16)*2**8).astype("uint8")
        img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)
        #plt.imshow(img_bf)
        #plt.show()

        return img_bf
    
    def click_event(self, event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            print(event, x, y)
            
            if (self.n_clicks == 0) & (self.round == "cells"):
                self.pts_dict[self.cell_object_num] = []
            elif (self.n_clicks == 0) & (self.round == "protrusion"):
                for probe in self.prev_prot.keys():
                    row = self.pts_dict[probe][0]
                    if 10 > np.sqrt((x-row[0][0])**2 + (y-row[0][1])**2):
                        self.spheroid_object_num = int(probe)
                
                self.pts_dict[self.spheroid_object_num] = []

            self.n_clicks += 1 
            self.pts.append([int(x*self.converter),int(y*self.converter), self.z_start, self.t_start, self.k])

            if (self.n_clicks == 1) & (self.round == "protrusion"):
                self.img_moc = cv2.circle(self.img_moc, self.pts[0][:2], radius=10, color=(0, 0, 255), thickness=4)
            elif (self.n_clicks == 2) & (self.round == "protrusion"):
                self.img_moc = cv2.arrowedLine(self.img_moc, self.pts[0][:2], self.pts[1][:2], (0, 0, 255)  , 5)
            elif (self.n_clicks == 3) & (self.round == "protrusion"):
                self.img_moc = cv2.circle(self.img_moc, (self.pts[2][0],self.pts[2][1]), radius=5, color=(255, 0, 255), thickness=-1)
            elif (self.n_clicks == 4) & (self.round == "protrusion"):
                self.img_moc = cv2.line(self.img_moc, self.pts[2][:2], self.pts[3][:2], (255, 0, 255) , 5)
                self.pts_dict[self.spheroid_object_num].append(self.pts)
                self.pts = []
                self.spheroid_object_num += 1
                self.n_clicks = 0

            elif (self.n_clicks == 1) & (self.round == "cells"):
                self.img_moc = cv2.circle(self.img_moc, self.pts[0][:2], radius=5, color=(0, 255, 255), thickness=-1)
                self.pts_dict[self.cell_object_num].append(self.pts)
                self.pts = []
                self.cell_object_num += 1
                self.n_clicks = 0

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.n_clicks = 0
            if len(self.pts) == 0:
                last_key = list(self.pts_dict)[-1]
                removed_tuple = self.pts_dict.pop(last_key)
            else:
                self.pts = []
                
            self.img_moc = self.img.copy()
            self.re_show()

        cv2.imshow("window", cv2.resize(self.img_moc, (self.scaled_size,self.scaled_size)) )

    
    def re_show(self):

        for current_key in self.pts_dict.keys():

            pts = self.pts_dict[current_key]
            if len(pts) == 1:
                print(pts)
                self.img_moc = cv2.circle(self.img_moc, pts[0][0][:2], radius=5, color=(0, 255, 255), thickness=-1)
            elif len(pts) == 4:
                self.img_moc = cv2.circle(self.img_moc, pts[0][0][:2], radius=10, color=(0, 0, 255), thickness=4)
                self.img_moc = cv2.arrowedLine(self.img_moc, pts[0][0][:2], pts[0][1][:2], (0, 0, 255)  , 9)
                self.img_moc = cv2.circle(self.img_moc, (pts[0][2][0], pts[0][2][1]), radius=5, color=(255, 0, 255), thickness=-1)
                self.img_moc = cv2.line(self.img_moc, pts[0][2][:2], pts[0][3][:2], (255, 255, 0)  , 9)

        windowText = "method {}, t={}, z={}, v={}".format(self.round , self.t_start, self.z_start, self.k)
        cv2.putText(self.img_moc, windowText,(150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        

    def process(self):

        # create a window
        cv2.namedWindow('window')
        # bind the callback function to window
        cv2.setMouseCallback('window', self.click_event)

        for video_path in tqdm.tqdm(self.target_paths, total=len(self.target_paths)):
            self.n_start = 0
            self.t_start = 0
            self.z_start = 0
            self.video_name = os.path.split(video_path)[-1][:-4]
            self.root_path = os.path.split(video_path)[0]
            self.results = os.path.join(self.root_path, "results_{}".format(self.video_name))
            self.saver = Cells(self.results)
            self.saver.reset_frame()

            _ = self.check_logged()

            parts = os.path.split(video_path)[-1].split("_")
            day = str(parts[0])        

            if day not in self.own_meta.keys():
                print(day, "Not in keys, skipping")
                continue

            with ND2Reader(video_path) as images:

                self.metas = load_metadata(images)
                if self.metas["n_channels"] == 2:
                    self.FL_flag = True
                else:
                    self.FL_flag = False

                self.idx_bf = 0
                if self.FL_flag:
                    for d in range(len(self.metas["channels"])):
                        if self.metas["channels"][d] == 'BF':
                            self.idx_bf = d
                        elif self.metas["channels"][d] == 'Red':
                            self.idx_fl = d
            
                
                for k in range(self.n_start, self.metas["n_fields"]):
                    self.k = k
                    t_cap = True

                    self.cell_object_num = 0
                    self.spheroid_object_num = 0
                    self.prev_prot = {}

                    while t_cap:

                        for stage in ["cells", "protrusion"]:

                            choosing = True
                            self.round = stage
                            self.n_clicks = 0
                            self.pts = []
                            self.object_num = 0
                            self.pts_dict = {}

                            # put coordinates as text on the image
                            self.img = self.fetch_image(images, self.t_start, self.z_start, k)
                            self.img_moc = self.img.copy()
                            windowText = "method {}, t={}, z={}, v={}".format(self.round , self.t_start, self.z_start, self.k)
                            cv2.putText(self.img_moc, windowText,(150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

                            while choosing:


                                cv2.imshow("window", cv2.resize(self.img_moc, (self.scaled_size,self.scaled_size)) )

                                # add wait key. window waits until user presses a key
                                kk = cv2.waitKey(0)
                                # and finally destroy/close all open windows

                                if kk == 113: #Exit q
                                    choosing = False
                                    print("exisiting methods")
                                    if (self.round == "cells") & (len(self.pts_dict.keys()) > 0):
                                        for current_key in self.pts_dict.keys():
                                            row = self.pts_dict[current_key][0]
                                            #print(row)
                                            #print(int(current_key), row[0][3], row[0][0], row[0][1], row[0][2],row[0][4])
                                            self.saver.update_cell(int(current_key), row[0][3], row[0][0], row[0][1], row[0][2],row[0][4])
                                    elif(self.round == "protrusion") & (len(self.pts_dict.keys()) > 0):
                                        for current_key in self.pts_dict.keys():
                                            row = self.pts_dict[current_key][0]
                                            #print(row)
                                            #print([[row[0][0], row[0][1]],[row[1][0], row[1][1]]], int(current_key), row[0][3], row[0][2],row[0][4], [[row[2][0], row[2][1]],[row[3][0], row[3][1]]])
                                            self.saver.update_vector([[row[0][0], row[0][1]],[row[1][0], row[1][1]]], int(current_key), row[0][3], row[0][2],row[0][4], [[row[2][0], row[2][1]],[row[3][0], row[3][1]]])

                                        self.prev_prot = self.pts_dict
                                elif kk == 101: #clear e
                                    self.n_clicks = 0
                                    self.pts = []
                                    self.object_num = 0
                                    self.pts_dict = {}
                                    self.img_moc = self.img.copy()
                                    windowText = "method {}, t={}/{}, z={}/{}, v={}/{}".format(self.round , self.t_start, self.metas["n_frames"], self.z_start, self.metas["n_levels"], self.k, self.metas["n_fields"])
                                    cv2.putText(self.img_moc, windowText,(150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                                elif kk == 119: #Move z up w
                                    self.z_start += 1
                                    if self.z_start == self.metas["n_levels"]:
                                        self.z_start -=1

                                    self.img = self.fetch_image(images, self.t_start, self.z_start, k)
                                    self.img_moc = self.img.copy()
                                    self.re_show()

                                elif kk == 115: #Move z down s
                                    self.z_start -= 1
                                    if self.z_start == -1:
                                        self.z_start +=1

                                    self.img = self.fetch_image(images, self.t_start, self.z_start, k)
                                    self.img_moc = self.img.copy()
                                    self.re_show()

                                elif kk == 100: #Move up d
                                    self.t_start += 1
                                    if self.t_start == self.metas["n_frames"]:
                                        self.t_start -=1

                                    self.img = self.fetch_image(images, self.t_start, self.z_start, k)
                                    self.img_moc = self.img.copy()
                                    self.re_show()

                                elif kk == 97: #Move down a
                                    self.t_start -= 1
                                    if self.t_start == -1:
                                        self.t_start +=1

                                    self.img = self.fetch_image(images, self.t_start, self.z_start, k)
                                    self.img_moc = self.img.copy()
                                    self.re_show()
                                elif kk == 114:
                                    choosing = False
                                    t_cap = False
                                else:
                                    print("incorrect key", kk)

                            self.n_clicks = 0
                            self.pts = []
                            self.object_num = 0
                            self.pts_dict = {}

                        self.save_logged()
                        if self.t_start == self.metas["n_frames"]:
                            t_cap = False
                        else:
                            self.t_start += 1

            if kk == 114:
                break

                        


if __name__ == "__main__":

    tracker = manual_tracker()
    tracker.process()