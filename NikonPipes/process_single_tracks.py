import argparse
import pandas as pd
import numpy as np
import glob
import os
import json
import cv2
import pickle
from nd2reader import ND2Reader

from tools.MTT.track_manager import TrackManager
from tools.func import *

class track_main():

    def __init__(self, args):

        if args:
            self.target_paths = [args.path]
        else:
            self.target_paths = self.find_paths()

        self.tracker = TrackManager(min_count=5, max_count = 5, gating = 500)
        self.kk = None
        self.df_tot = pd.DataFrame(data=None, columns=["day", "location", "time", "x", "y", "z", "location"])


    def find_paths(self):
        paths_single_track = glob.glob("D:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv") 
        paths_single_track += glob.glob("F:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv")
        paths_single_track += glob.glob("G:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv") 
        paths_single_track += glob.glob("E:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv")
        paths_single_track += glob.glob("H:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv")

        return paths_single_track

    def download_accesories(self):

        video_name = os.path.split(self.video_path)[-1][:-4]
        root_path = os.path.split(self.video_path)[0]
        results = os.path.join(root_path, "results_{}".format(video_name))

        focus_path = glob.glob(os.path.join(results, "corrected_focus_indixes.pkl")) #*_indixes.pkl

        if len(focus_path) == 0:
            print("No focus correction!")
            focus_path = glob.glob(os.path.join(results, "focus_indixes.pkl"))

        with open(focus_path[0], 'rb') as f:
            self.focus_dict = pickle.load(f)   


    def pipe(self):
        cv2.namedWindow('window')

        for counter, file_path in enumerate(self.target_paths):

            print(file_path)
            vector_file = os.path.join(os.path.split(file_path)[0], "data_vector.csv")
            self.df_vector = pd.read_csv(vector_file)

            self.parts = os.path.split(os.path.split(file_path)[0])[1].split("_")
            self.day = self.parts[1]

            self.video_path = os.path.split(os.path.split(file_path)[0])[0]
            self.video_path = glob.glob(os.path.join(self.video_path,"{}_*".format(self.day)))[0]
            root_path = os.path.split(self.video_path)[0]
            video_name = os.path.split(self.video_path)[-1][:-4]
            self.results = os.path.join(root_path, "results_{}".format(video_name))

            self.download_accesories()

            with open('./dataStore/metalib.json', 'r') as f:
                self.own_meta = json.load(f)
            
            self.own_meta = self.own_meta[self.day]
            self.pixel_size = self.own_meta["m"]
            self.z_step = self.own_meta["z_level"]

            with ND2Reader(self.video_path) as images:
                try:
                    metas = load_metadata(images)
                except:
                    print("broken metafile", self.video_path)
                    continue


                if metas["n_channels"] == 2:
                    for d in range(len(metas["channels"])):
                        if metas["channels"][d] == 'BF':
                            self.idx_bf = d
                else:
                    self.idx_bf = 0

            df = pd.read_csv(file_path)


            for tags, data_location in df.groupby(["location"]):

                self.loc = tags[0]

                if self.loc < len(self.own_meta["cell"]):
                    self.line_name = self.own_meta["cell"][self.loc]
                else:
                    self.line_name = "unknown" 

                out_name = os.path.join(self.results,'{}_{}_{}_individual.mp4'.format(os.path.split(self.video_path)[1][:-4], (self.loc), (self.line_name) ) )
                self.out_process = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), 5, (2304,2304))
                data_location = data_location.reset_index(drop=True)

                self.process_location(data_location)
                self.out_process.release()
                if self.kk == 113:
                    break



    def process_location(self, df):
        
        for tags_, data_stamp in df.groupby("time"):
            try:
                focus_idx = self.focus_dict[self.loc][tags_]
            except:
                print("focus dict missing", tags_, "in location", self.loc)
                break

            if focus_idx == -1:
                break

            with ND2Reader(self.video_path) as images:

                try:
                    img_bf = images.get_frame_2D(c=self.idx_bf, t=tags_, z=focus_idx, x=0, y=0, v=self.loc)
                except:
                    img_bf = images.get_frame_2D(c=self.idx_bf, t=tags_-1, z=focus_idx, x=0, y=0, v=self.loc)

            img_bf = (img_bf/(2**16)*2**8).astype("uint8")
            self.vis_img = np.stack((img_bf, img_bf, img_bf), axis = -1)

            #self.vis_img = np.zeros((int(2304*self.own_meta["m"]),2304,3))

            df_vector_sub = self.df_vector[(self.df_vector["location"] == self.loc) & (self.df_vector["time"] == tags_)]
            values = df_vector_sub[["x", "y", "x2", "y2"]].values
            for i in range(values.shape[0]):
                self.vis_img = cv2.arrowedLine(self.vis_img, (values[i,0],values[i,1]),  (values[i,2],values[i,3]), (0, 0, 255)  , 5)

            data_stamp = data_stamp.reset_index(drop = True)

            data_stamp["labels"] = 1
            data_stamp["dummy"] = 1
            #data_stamp["x"] *= self.pixel_size; data_stamp["y"] *= self.pixel_size; data_stamp["z"] *= self.z_step
            dets = data_stamp[["x", "y", "z", "labels", "dummy"]].values

            self.tracker.update(dets, tags_, np.zeros(data_stamp.shape[0]))
            all_tracks = self.tracker.trackers

            for track in all_tracks:

                if (not track.killed):

                    t_col = (255,0,0)
                    m_col = (255,0,0)

                    if track.tentative:
                        t_col = (100,100,100)
                        m_col = (100,100,100)

                    track_info = (np.array(track.history[-5:])[:,:2]).reshape((-1, 1, 2))
                    
                    #if track_info.shape[0] > 1:


                    if (track_info[-1,0,0] != track_info[-1,0,1]):
                        self.vis_img = cv2.polylines(self.vis_img, np.int32([track_info]), True, t_col, 2)
                        self.vis_img = cv2.drawMarker(self.vis_img, (int(track_info[-1,0,0]),int(track_info[-1,0,1])), m_col, 0, 50, 5)
        
            windowText = "loc: {}, time: {}".format(self.loc, tags_)
            # put coordinates as text on the image
            cv2.putText(self.vis_img, windowText,(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
            cv2.imshow("window",cv2.resize(self.vis_img, (1024,1024)))
            self.out_process.write(self.vis_img)
            self.kk = cv2.waitKey(1)
            if self.kk == 113: #Exit 
                break


    def save_data(self):
        #self.longestTrack = np.max([np.max(i.indices) for i in self.pickleData.trackers])
        #df["time"] = np.arange(self.longestTrack)



        for i in np.arange(len(self.tracker.trackers)):

            df = pd.DataFrame(data=None, columns=["day", "location", "time", "x", "y", "z", "location"])

            #pick stamps of detections and coordinates
            self.dispData = np.array(self.tracker.trackers[i].history)
            self.time = np.array(self.tracker.trackers[i].indices)
            df["time"] = self.time

            df["x"]= self.dispData[:,0]*self.pixel_size
            df["y"]= self.dispData[:,1]*self.pixel_size
            df["z"]= self.dispData[:,2]*self.z_step
            df["location"] = i
            df["day"] = self.day

            self.df_tot = pd.concat((self.df_tot,df))

        print(self.df_tot)




if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
        description="""Download results in the folder and ouputs results
                    """)
    parser.add_argument('--path','-p',required=False,default= None, help='Path to folder. eg. C:/data/imgs')

    #Save arguments
    args = parser.parse_known_args()[0]

    process = track_main(args)
    process.pipe()


"""
            focus_path = glob.glob(os.path.join(results, "corrected_focus_indixes.pkl")) #*_indixes.pkl
            if len(focus_path) == 0:
                print("No focus correction!")
                focus_path = glob.glob(os.path.join(results, "focus_indixes.pkl"))
            with open(focus_path[0], 'rb') as f:
                self.focus_dict = pickle.load(f)   

            pickel_path = glob.glob(os.path.join(results,"{}_corrected_detections.pkl".format(video_name)))

            if len(pickel_path) == 0:
                print("No pkl corrected file!")
                pickel_path = os.path.join(results,"{}_detections.pkl".format(video_name))

            with open(pickel_path, 'rb') as f:
                self.data_dict = pickle.load(f)
"""