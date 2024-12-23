import argparse
import pandas as pd
import numpy as np
import glob
import os
import json
import cv2
import pickle
from nd2reader import ND2Reader

from tools.MTT_o.track_manager import TrackManager
#from tools.cMTT.track_manager import TrackManager
from tools.func import *

class track_main():

    def __init__(self, args):

        if args.path:
            self.target_paths = [args.path]
        else:
            self.target_paths = self.find_paths()

        self.gen = args.gen
        self.tracker  = None

        self.init_tracker()

        self.kk = None 


    def init_tracker(self):
        #self.tracker = TrackManager(min_count=5, max_count = 6, gating = 500)
        self.tracker  = TrackManager(min_count=5,max_count=6, gating_spawn = 150, gating_far= 100)
            

    def find_paths(self):
        paths_single_track = glob.glob("D:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv") 
        paths_single_track += glob.glob("F:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv*")
        paths_single_track += glob.glob("G:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv*") 
        paths_single_track += glob.glob("E:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv*")
        paths_single_track += glob.glob("H:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv*")
        paths_single_track += glob.glob("I:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv*")
        paths_single_track += glob.glob("J:/instru_projects/TimeLapses/u-wells/*/*/data_track.csv*")
        #paths_single_track = glob.glob("E:/instru_projects/TimeLapses/u-wells/IPN/*/detector_track.csv") 

        
        return paths_single_track

    def download_accesories(self):

        focus_path = glob.glob(os.path.join(self.results, "corrected_focus_indixes.pkl")) #*_indixes.pkl

        if len(focus_path) == 0:
            print("No focus correction!")
            focus_path = glob.glob(os.path.join(self.results, "focus_indixes.pkl"))

        with open(focus_path[0], 'rb') as f:
            self.focus_dict = pickle.load(f) 

        with open('./dataStore/metalib.json', 'r') as f:
            self.own_meta_ = json.load(f)


    def process_location(self, df):
        
        for tags_, data_stamp in df.groupby("time"):
            tags_ = int(tags_)

            data_stamp = data_stamp.reset_index(drop = True)
            data_stamp["labels"] = 1
            data_stamp["dummy"] = 1

            try:
                focus_idx = self.focus_dict[self.loc][tags_]
            except:
                print("focus dict missing", tags_, "in location", self.loc)
                break

            if (focus_idx == -1) | (focus_idx == -2):
                print("Analysis interupted due to drifting or overgrowth.")
                break

            df_vector_sub = self.df_vector[(self.df_vector["location"] == self.loc) & (self.df_vector["time"] == tags_)]
            protrusion_t = df_vector_sub[["x", "y", "x2", "y2"]].values

            if self.gen:
                with ND2Reader(self.video_path) as images:
                    try:
                        img_bf = images.get_frame_2D(c=self.idx_bf, t=tags_, z=focus_idx, x=0, y=0, v=self.loc)
                    except:
                        img_bf = images.get_frame_2D(c=self.idx_bf, t=tags_-1, z=focus_idx, x=0, y=0, v=self.loc)

                img_bf = (img_bf/(2**16)*2**8).astype("uint8")
                self.vis_img = np.stack((img_bf, img_bf, img_bf), axis = -1)

                for i in range(protrusion_t.shape[0]):
                    self.vis_img = cv2.arrowedLine(self.vis_img, (protrusion_t[i,0],protrusion_t[i,1]),  (protrusion_t[i,2],protrusion_t[i,3]), (0, 0, 255)  , 5)


            dets = data_stamp[["x", "y", "z", "labels", "dummy"]].values
            self.tracker.update(dets, tags_) 

            all_tracks = self.tracker.trackers

            if self.gen:
                for track in all_tracks:

                    if (not track.killed):

                        t_col = (255,0,0)
                        m_col = (255,0,0)

                        if track.tentative:
                            t_col = (100,100,100)
                            m_col = (100,100,100)

                        track_info = (np.array(track.history[-5:])[:,:2]).reshape((-1, 1, 2))

                        if ((track_info[-1,0,0] != track_info[-1,0,1]) ):
                            if track.tentative:
                                continue

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

    def pipe(self):

        if self.gen:
            cv2.namedWindow('window')

        for counter, file_path in enumerate(self.target_paths):

            print("processing: ", file_path, "\n Done: ", counter, len(self.target_paths)-1)
            
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
            
            self.own_meta = self.own_meta_[self.day]
            self.pixel_size = self.own_meta["m"]
            self.z_step = self.own_meta["z_level"]

            with ND2Reader(self.video_path) as images:

                try:
                    metas = load_metadata(images)

                    if metas["n_channels"] == 2:
                        for d in range(len(metas["channels"])):
                            if metas["channels"][d] == 'BF':
                                self.idx_bf = d
                    else:
                        self.idx_bf = 0
                except:
                    print("broken metafile", self.video_path)
                    if self.day == "240520":
                        self.metas = { "n_fields": 7, "n_frames": 25, "n_levels": 27}
                        self.idx_bf = 0
                        self.idx_fl = 0
                    else:
                        print("Cannot open meta from ", file_path, " quiting...")
                        exit()

            df = pd.read_csv(file_path) 

            self.df_tot_single = []#pd.DataFrame()
            self.df_tot_prot = []#pd.DataFrame()

            for tags, data_location in df.groupby(["location"]):

                if self.gen:
                    out_name = os.path.join(self.results,'{}_{}_{}_individual.mp4'.format(os.path.split(self.video_path)[1][:-4], (self.loc), (self.line_name) ) )
                    self.out_process = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), 5, (2304,2304))

                self.init_tracker()
                self.track_ID = 0



                self.loc = int(tags[0])

                if self.loc < len(self.own_meta["cell"]):
                    self.line_name = self.own_meta["cell"][self.loc]
                else:
                    print("Size of metafile dictionary does not match with data.")
                    exit()
                
                data_location = data_location.reset_index(drop=True)
                self.process_location(data_location)

                if self.gen:
                    self.out_process.release()

                if self.kk == 113:
                    break

                self.save_single()

            if self.df_tot_single:

                self.df_tot_single_ = pd.concat(self.df_tot_single)
                save_path = os.path.join(self.results, "data_tracks_results.csv")
                self.df_tot_single_.to_csv(save_path, index = False)
            
            self.save_protrusion()



    def save_protrusion(self):

        df_ids = pd.read_csv("./dataStore/ExpDesign2_.csv")
        #self.df_tot_prot  = []
        for label_info, sub_data in self.df_vector.groupby(["cell_id", "location"]):

            loc = int(label_info[1])
            id = int(label_info[0])

            df_info = df_ids[(df_ids["day"] == int(self.day))& (df_ids["location"] == int(loc)) ]#.reset_index(drop=True) #& (df_ids["location"] == i)

            if df_info.shape[0] == 0:
                print("saving, did not find location from, loc", int(self.day), ",from location ",loc)
                continue
                
            sub_data = sub_data.reset_index(drop = True)
            try:
                sub_data = sub_data.iloc[sub_data['lenght'].idxmax()]
            except:
                sub_data = sub_data.iloc[sub_data['length'].idxmax()]
                
            sub_data = sub_data.to_frame().T
            
            sub_data["day"] = self.day

            sub_data["ID_running"] = df_info["ID_running"].values[0]
            sub_data["time"] = int(sub_data["time"])*self.own_meta["dt"]/60**2+self.own_meta["incubation_time"]
            sub_data["cell_label"] = df_info['cell_label'].values[0]
            sub_data["well_id"] = df_info['well_id'].values[0]
            sub_data["measurement_id"] = df_info['measurement_id'].values[0]
            sub_data["matrix"] = df_info['matrix'].values[0]
            sub_data["protrusion_ID"] = id

            self.df_tot_prot.append(sub_data)

        if len(self.df_tot_prot):
            self.df_tot_prot_ = pd.concat(self.df_tot_prot)
            save_path = os.path.join(self.results, "data_vector_results.csv")
            self.df_tot_prot_.to_csv(save_path, index = False)



    def save_single(self):
        
        #self.longestTrack = np.max([np.max(i.indices) for i in self.pickleData.trackers])
        #df["time"] = np.arange(self.longestTrack)

        df_ids = pd.read_csv("./dataStore/ExpDesign2_.csv")

        df_info = df_ids[(df_ids["day"] == int(self.day)) & (df_ids["location"] == int(self.loc))]
        if df_info.shape[0] == 0:
            print("save single, did not find location, loc", int(self.day), self.loc)
        else:
            for i in np.arange(len(self.tracker.trackers_all)):

                if 1 not in self.tracker.trackers_all[i].status:
                    #print("No confirmed detections from the targer", int(self.day), "/",self.loc, "/",i)
                    #print(self.tracker.trackers_all[i].status)
                    continue

                df = pd.DataFrame() #data=None, columns=["day", "location", "time", "x", "y", "z", "location"]

                #pick stamps of detections and coordinates

                self.dispData = np.array(self.tracker.trackers_all[i].history)
                self.time = np.array(self.tracker.trackers_all[i].indices)

                df["x"]= self.dispData[:,0]*self.pixel_size
                df["y"]= self.dispData[:,1]*self.pixel_size
                df["z"]= self.dispData[:,2]*self.z_step

                df["time"] = self.time*self.own_meta["dt"]/60**2+self.own_meta["incubation_time"]
                df["location"] = self.loc
                df["day"] = self.day
                df["track_label"] = self.track_ID
                self.track_ID += 1
                
                df["ID_running"] = df_info["ID_running"].values[0]
                df["cell_label"] = df_info['cell_label'].values[0]
                df["well_id"] = df_info['well_id'].values[0]
                df["measurement_id"] = df_info['measurement_id'].values[0]
                df["matrix"] = df_info['matrix'].values[0]

                self.df_tot_single.append(df)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
        description="""Download results in the folder and ouputs results
                    """)
    parser.add_argument('--path','-p',required=False,default= None, help='Path to folder. eg. C:/data/imgs')
    parser.add_argument('--gen','-g', required=False, default= False, type = bool,  help='Generate videos')

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