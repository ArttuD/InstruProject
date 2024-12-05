
import os
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 
import glob

import tqdm

from nd2reader import ND2Reader
import h5py
import pickle
import json

import datetime

from skimage.filters import rank, threshold_otsu, threshold_local#
from skimage import morphology

import plotly


import scipy 
import skimage

import pandas as pd
import seaborn as sns
import csv
from sklearn.cluster import DBSCAN

from tools.func import *
import plotly
import plotly.express as px

# %%


class Collect_Data():

    def __init__(self):
    
        self.cnt_limit = 6
        self.meta_columns = ["seeding_density", "cell_label", "well_id", "measurement_id", "matrix", "ID", "day"]

        with open('./dataStore/metalib.json', 'r') as f:
            self.own_meta = json.load(f)

        self.collect_folders()

        #self.spheroid_info = pd.read_csv('./dataStore/ExpDesign2_.csv')  
        #with open(r"./dataStore/ExpDesig2ID_.pkl", "rb") as input_file:
        #    self.cntDict = pickle.load(input_file)

    def collect_folders(self):

        self.target_paths = glob.glob("D:/instru_projects/TimeLapses/u-wells/*/*.nd2") 
        self.target_paths += glob.glob("F:/instru_projects/TimeLapses/u-wells/*/*.nd2")
        self.target_paths += glob.glob("G:/instru_projects/TimeLapses/u-wells/*/*.nd2") 
        self.target_paths += glob.glob("E:/instru_projects/TimeLapses/u-wells/*/*.nd2")
        self.target_paths += glob.glob("H:/instru_projects/TimeLapses/u-wells/*/*.nd2")
        self.target_paths += glob.glob("I:/instru_projects/TimeLapses/u-wells/*/*.nd2")
        self.target_paths += glob.glob("J:/instru_projects/TimeLapses/u-wells/*/*.nd2")

        self.paths_single_track = glob.glob("D:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv") 
        self.paths_single_track += glob.glob("F:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv")
        self.paths_single_track += glob.glob("G:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv") 
        self.paths_single_track += glob.glob("E:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv")
        self.paths_single_track += glob.glob("H:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv")
        self.paths_single_track += glob.glob("I:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv")
        self.paths_single_track += glob.glob("J:/instru_projects/TimeLapses/u-wells/*/*/data_tracks_results.csv")

        self.paths_single_vector = glob.glob("D:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv") 
        self.paths_single_vector += glob.glob("F:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv")
        self.paths_single_vector += glob.glob("G:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv") 
        self.paths_single_vector += glob.glob("E:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv")
        self.paths_single_vector += glob.glob("H:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv") 
        self.paths_single_vector += glob.glob("I:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv")
        self.paths_single_vector += glob.glob("J:/instru_projects/TimeLapses/u-wells/*/*/data_vector_results.csv")

    def worker(self):

        self.matched_measurements = self.match_mesurements()
        self.spheroid_info, self.cntDict  = self.parse_spheroids()

    def generate_plots(self):
        self.single_info = self.parse_single_tracks()
        self.vector_info = self.process_vector()


    def match_mesurements(self):

        df_tot = []
        processed_list = []

        meas_run = 0

        for video_path in self.target_paths:

            if video_path in processed_list:
                continue

            parts = os.path.split(video_path)[-1].split("_")
            day = str(parts[0])

            coord_list_x, coord_list_y, coord_list_z = self.fetch_coords_well(video_path)

            df = pd.DataFrame(np.stack((coord_list_x, coord_list_y, coord_list_z), axis = -1), columns = ["x", "y", "z"])
            init_length = df.shape[0]

            df["day"] = day
            df["well_order"] = np.arange(df.shape[0])

            search = 1
            last_let = -100

            while search>0:

                if last_let == -100:
                    last_let = int(day[-1])

                if search == 1:
                    last_let-=1
                elif search == 2:
                    last_let+=1
                    
                day_probe = list(day)
                if last_let != -1:
                    day_probe[-1] = str(last_let)
                else:
                    day_probe[-1] = str(9)
                    day_probe[-2] = str(int(day_probe[-2])-1)

                day_probe = "".join(day_probe)
                if day_probe == "240521":
                    day_probe = "240522"

                parts_ = day_probe + "*.nd2"
                parts_ = "".join(parts_)

                found_flag = False

                #What if 
                for letter in ["F", "G", "D", "E", "H", "I", "J"] :#

                    probe_path = os.path.join(os.path.split(video_path)[0], parts_)
                    probe_path = list(probe_path)
                    probe_path[0] = letter
                    probe_path = "".join(probe_path)
                    found_paths = glob.glob(probe_path)

                    if len(found_paths) > 0:
                        found_flag = True
                        break

                if (found_flag):
                    if (found_paths[0] not in processed_list):
                        #print("Found potential path:", found_paths)
                        coord_list_x_, coord_list_y_, coord_list_z_ = self.fetch_coords_well(found_paths[0])

                        df_ = pd.DataFrame(np.stack((coord_list_x_, coord_list_y_, coord_list_z_), axis = -1), columns = ["x", "y", "z"])
                        df_["day"] = day_probe
                        df_["well_order"] = np.arange(df_.shape[0])
                        df = pd.concat((df,df_)).reset_index(drop=True)

                        processed_list.append(found_paths[0])
                        found_flag = False
                    else:
                        found_flag = False
                        search += 1
                        last_let = -100

                        if search == 3:
                            print("Measurement set done. Combined: ", df["day"].unique())
                            search = 0                      
                else:
                    found_flag = False
                    search += 1
                    last_let = -100

                    if search == 3:
                        print("Measurement set done. Combined: ", df["day"].unique())
                        search = 0

            if df.shape == init_length:
                df["well_id"] = np.arange(init_length)
                df["measurement_id"] = meas_run
            else: 
                clustering = DBSCAN(eps=1400, min_samples=1).fit(df[["x", "y"]].values)
                df["well_id"] = clustering.labels_
                df["measurement_id"] = meas_run

            #fig = px.scatter_3d(df, x='x', y='y', z='z', color='well_id')
            #plotly.offline.plot(fig, filename=os.path.join('./dataStore', "mathced_{}.html".format(meas_run)))
            #fig.close()

            meas_run += 1
            df_tot.append(df)

        df_tot = pd.concat(df_tot)

        return df_tot


    def fetch_coords_well(self, video_path):
        fh = video_path
        label_map = None

        if isinstance(fh, str):
            if not fh.endswith(".nd2"):
                raise InvalidFileType(
                    ("The file %s you want to read with nd2reader" % fh)
                    + " does not have extension .nd2."
                )
            
            filename = fh

            fh = open(fh, "rb")

        _fh = fh
        _fh.seek(-8, 2)
        chunk_map_start_location = struct.unpack("Q", _fh.read(8))[0]
        _fh.seek(chunk_map_start_location)
        raw_text = _fh.read(-1)
        label_map = LabelMap(raw_text)
        datasTT = RawMetadata(_fh, label_map)

        coord_list_x = []
        coord_list_y = []
        coord_list_z = []

        well_info = datasTT.image_metadata[b'SLxExperiment'][b'ppNextLevelEx'][b''][b'uLoopPars'][b'Points'][b'']

        for i in range(len(well_info)):
            
            coord_y = (well_info[i][b'dPosY'])
            coord_x = (well_info[i][b'dPosX'])
            coord_z = (well_info[i][b'dPosZ'])

            coord_list_x.append(coord_x); coord_list_y.append(coord_y); coord_list_z.append(coord_z)
        
        return coord_list_x, coord_list_y, coord_list_z
    
    def parse_spheroids(self):
        
        total_ID = 0
        contours_dict = {}
        df_lists = []

        for global_counter, video_path in enumerate(self.target_paths):

            print("Processing: ", global_counter, "/", len(self.target_paths))

            video_name = os.path.split(video_path)[-1][:-4]
            results = os.path.join(os.path.split(video_path)[0], "results_{}".format(video_name))
            pickel_path = os.path.join(results,"{}_corrected_detections.pkl".format(video_name))

            if len(pickel_path) == 0:
                print("No corrected pickel!")
                pickel_path = os.path.join(results, "{}_detections.pkl".format(video_name))

            if os.path.isfile(pickel_path):
                with open(pickel_path, 'rb') as f:
                    total_dict = pickle.load(f)

            parts = os.path.split(video_path)[-1].split("_")
            day = str(parts[0])

            if day not in self.own_meta.keys():
                print("passing day {}, form file {} because not recorded in metadata.json".format(day, video_path))
                exit()

            if "m" not in self.own_meta[day].keys():
                print("Missing pixel size from ", video_path)
                exit()

            focus_path = glob.glob(os.path.join(results, "corrected_focus_indixes.pkl")) #*_indixes.pkl

            if len(focus_path) == 0:
                print("No corrected focus correction!")
                focus_path = glob.glob(os.path.join(results, "focus_indixes.pkl"))

            with open(focus_path[0], 'rb') as f:
                focus_dict = pickle.load(f)

            for counter, current_key in enumerate(total_dict.keys()):

                if len(total_dict[current_key]["mask"]) < self.cnt_limit:
                    print("Day {} location {} too short, only {} frames".format(day, current_key, len(total_dict[current_key]["mask"])))
                    continue

                loc_ = int(current_key.split("_")[1])

                if (loc_ in self.own_meta[day]["ignore"]) | (loc_ in self.own_meta[day]["multi"]):
                    print("Skipping {}, loc {} bcs set ignore or multiple: ".format(day, loc_))
                    continue

                contours_dict[total_ID] = {} 
                contours_dict[total_ID]["masks"] = []
                contours_dict[total_ID]["x"] = []
                contours_dict[total_ID]["y"] = []
                contours_dict[total_ID]["z"] = []
                contours_dict[total_ID]["time"] = []
                contours_dict[total_ID]["area"] = []

                for n in range(len(total_dict[current_key]['mask'])):

                    idx = int(focus_dict[loc_][n])
                    if (idx == -1) | (idx == -2):
                        print("Overgrown or drifted, moving to next")
                        break

                    cnt = total_dict[current_key]['mask'][n][total_dict[current_key]["big_idx"][n]]
                    contours_dict[total_ID]["masks"].append(cnt)
                    
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    area = cv2.contourArea(cnt)

                    contours_dict[total_ID]["x"].append(x*self.own_meta[day]["m"])
                    contours_dict[total_ID]["y"].append(y*self.own_meta[day]["m"])
                    contours_dict[total_ID]["z"].append(idx*self.own_meta[day]["z_step"])

                    contours_dict[total_ID]["time"].append(n*self.own_meta[day]["dt"]/60**2+self.own_meta[day]["incubation_time"])
                    contours_dict[total_ID]["area"].append(area*self.own_meta[day]["m"]**2)

                
                df_temp = pd.DataFrame(np.zeros((1,len(self.meta_columns))), columns=self.meta_columns)
                cell_data = self.own_meta[day]["cell"][loc_]
                
                if isinstance(cell_data, list) and len(cell_data) == 2:
                    df_temp["cell_label"] = str(cell_data[0])  # cell_index1 -> cell_label
                    df_temp["matrix"] = str(cell_data[1])      # cell_index2 -> matrix
                else:
                    df_temp["cell_label"] = str(self.own_meta[day]["cell"][loc_])
                    df_temp["matrix"] = str(self.own_meta[day]["matrix"])

                df_temp["seeding_density"] = int(self.own_meta[day]["seeding_density"][loc_])
                df_temp["well_id"] = self.matched_measurements[(self.matched_measurements["day"] == day) & (self.matched_measurements["well_order"] == loc_ )]["well_id"].values[0]
                df_temp["measurement_id"] = self.matched_measurements[(self.matched_measurements["day"] == day) & (self.matched_measurements["well_order"] == loc_)]["measurement_id"].values[0]


                df_temp["ID_running"] = total_ID
                df_temp["day"] = day
                df_temp["location"] = loc_

                total_ID += 1

                if len(df_temp) == 0:
                    print("Not saving", loc_, day)
                    total_ID -= 1
                    pass
                else:
                    df_lists.append(df_temp)

            df_temp = pd.concat(df_lists)

            df_temp.loc[df_temp["cell_label"] == "T", "cell_label"] = "MCF10AT"
            df_temp.loc[df_temp["cell_label"] == "MFC10AT", "cell_label"] = "MCF10AT"
            df_temp.loc[df_temp["cell_label"] == "MCFA10A", "cell_label"] = "MCF10A"
            df_temp.loc[df_temp["cell_label"] == "MCFA10", "cell_label"] = "MCF10A"
            df_temp.loc[df_temp["cell_label"] == "DCIS", "cell_label"] = "DCIS.COM"
            df_temp.loc[df_temp["cell_label"] == "DCIS.COM", "cell_label"] = "MCF10DCIS.com"
            df_temp.loc[df_temp["matrix"] == "IPN3", "matrix"] = "IPN3mM"
            df_temp.loc[df_temp["matrix"] == "IPN22", "matrix"] = "IPN22mM"
            df_temp.loc[df_temp["matrix"] == "22mM", "matrix"] = "IPN22mM"
            df_temp.loc[df_temp["matrix"] == "coll", "matrix"] = "collagen"
            df_temp.loc[df_temp["matrix"] == "2mgml", "matrix"] = "collagen"

            df_temp.to_csv('./dataStore/ExpDesign2_.csv', index=False) 
            with open("./dataStore/ExpDesig2ID_.pkl", 'wb') as f:
                pickle.dump(contours_dict, f)
            
        df_all = pd.concat(df_lists)

        df_all.loc[df_all["cell_label"] == "T", "cell_label"] = "MCF10AT"
        df_all.loc[df_all["cell_label"] == "MFC10AT", "cell_label"] = "MCF10AT"
        df_all.loc[df_all["cell_label"] == "MCFA10A", "cell_label"] = "MCF10A"
        df_all.loc[df_all["cell_label"] == "MCFA10", "cell_label"] = "MCF10A"
        df_all.loc[df_all["cell_label"] == "DCIS", "cell_label"] = "DCIS.COM"
        df_all.loc[df_all["cell_label"] == "DCIS.COM", "cell_label"] = "MCF10DCIS.com"
        df_all.loc[df_all["matrix"] == "IPN3", "matrix"] = "IPN3mM"
        df_all.loc[df_all["matrix"] == "IPN22", "matrix"] = "IPN22mM"
        df_all.loc[df_all["matrix"] == "22mM", "matrix"] = "IPN22mM"
        df_all.loc[df_all["matrix"] == "coll", "matrix"] = "collagen"
        df_all.loc[df_all["matrix"] == "2mgml", "matrix"] = "collagen"

        df_all.to_csv('./dataStore/ExpDesign2_.csv', index=False) 
        with open("./dataStore/ExpDesig2ID_.pkl", 'wb') as f:
            pickle.dump(contours_dict, f)

        return df_all, contours_dict
        
    def save(self):
    
        self.spheroid_info.to_csv('./dataStore/ExpDesign2_.csv', index=False)  
        self.single_info.to_csv('./dataStore/ExpDesign2_single_.csv', index=False)  
        self.vector_info.to_csv('./dataStore/ExpDesign2_vector_.csv', index=False)   

        with open("./dataStore/ExpDesig2ID_.pkl", 'wb') as f:
            pickle.dump(self.cntDict, f)


    def parse_single_tracks(self):

        df_sp = []

        #video_path = paths_single_track[0]
        for global_counter, video_path in enumerate(self.paths_single_track):

            df_single = pd.read_csv(video_path)

            for tags, data in df_single.groupby(["measurement_id","well_id"]):

                data = data.reset_index(drop=True)
                df_sp.append(self.parse_single(data))

            print("Cells processed: ",global_counter,"/",len(self.paths_single_track)-1)
            pd.concat(df_sp).to_csv('./dataStore/ExpDesign2_single_.csv', index=False)  

        df = pd.concat(df_sp)

        return df
            

    def parse_id(self, df, id_counter):

        prev = -1
        drop_list = []
        for i in range(df.shape[0]):
            
            if df["time"].values[i] > prev:
                df.loc[i, "cell_id"] = id_counter
                prev = df["time"].values[i]
                
            elif (df["time"].values[i] == prev):
                drop_list.append(i)
                print("double", i)
            else:
                id_counter += 1
                df.loc[i, "cell_id"] = id_counter
                prev = df["time"].values[i]
            
        return df, id_counter, drop_list


    def parse_cnt_dict(self, num_key):

        #for count, ckey in enumerate(num_key):

        currentDict = self.cntDict[num_key]

        try:
            del currentDict["masks"]
        except:
            pass

        df_spheroid = pd.DataFrame.from_dict(currentDict)

        df_spheroid["x"] = filter_Gauss(df_spheroid["x"].values)
        df_spheroid["y"] = filter_Gauss(df_spheroid["y"].values)
        df_spheroid["z"] = filter_Gauss(df_spheroid["z"].values)
        df_spheroid["area"] = filter_Gauss(df_spheroid["area"].values)

        return df_spheroid


    def calc_MSD_(self, data, t, tau):
        
        x = data["x"] - data["x"].values[0]
        y = data["y"] - data["y"].values[0]

        MSD = np.zeros(len(x))

        for idx in np.arange(len(x)):
            if idx+tau >= len(x):
                break

            MSD[idx]= np.sqrt( ((x[idx+tau]-x[idx])/(t[idx+tau]-t[idx]))**2 + ((y[idx+tau]-y[idx])/(t[idx+tau]-t[idx]))**2)
                
        
        return MSD[:idx].mean()

    def parse_single(self, data):

        data["MSD_tau"] = np.zeros(data.shape[0])
        data["t"] = np.zeros(data.shape[0])

        for tau in range(1, data.shape[0]):

            MSD = self.calc_MSD_(data ,data["time"].values , tau)
            data.loc[tau,"MSD_tau"] = MSD
            data.loc[tau,"t"] = np.mean(data["time"].values[::tau])

        try:
            p, p_ = scipy.optimize.curve_fit(power_law, data["t"], data["MSD_tau"],  maxfev = 10000)
            data["alpha"] = p[1]
            data["ampltiude"] = p[0]
        except:
            data["alpha"] = 0 #np.nan
            data["ampltiude"] = 0 #np.nan

        df = []

        for count, cluster in enumerate(data.groupby(["ID_running"])):

            single_tags = cluster[0]
            single_data = cluster[1]

            single_data = single_data.reset_index(drop=True)
            single_tags = int(single_tags[0])

            spheroid_df = self.parse_cnt_dict(single_tags)

            if spheroid_df.shape[0] == 0:
                print("No data recorded from spheroid spheroid with running ID", single_tags)
                continue

            idx = np.argmin(spheroid_df["time"].values - single_data["time"][0])
            
            try:
                sub_sub_df = spheroid_df[idx: (idx+single_data.shape[0])]
            except:
                print("Cells tracked incorrectly, cannot fetch location: ", single_tags)
                continue

            single_data["dx"] = 0; single_data["dy"] = 0; single_data["angle"] = 0

            single_data.loc[1:,"dx"] = np.diff(single_data["x"])
            single_data.loc[1:, "dy"] = np.diff(single_data["y"])

            v1 = np.stack((single_data["dx"], single_data["dy"]), axis = -1)
            v2 = np.stack((sub_sub_df["x"]-single_data["x"], sub_sub_df["y"]-single_data["y"]), axis = -1)

            angles_ = np.empty(v2.shape[0])
            
            for n in range(v2.shape[0]):
                angles_[n] = angle(v1[n,:], v2[n,:])

            deg_angles = np.rad2deg(angles_)
            #deg_angles = deg_angles[~np.isnan(deg_angles)]
            single_data.loc[:(v2.shape[0]-1), "angle"] = deg_angles
            
            df.append(single_data)
        

        df = pd.concat(df)

        return df


    def process_vector(self):

        df_vector = []

        #video_path = paths_single_track[0]
        for global_counter, video_path in enumerate(self.paths_single_vector):

            df_ = pd.read_csv(video_path)

            for tags, data in df_.groupby(["measurement_id","well_id"]):

                data = data.reset_index(drop=True)
                df_vector.append(self.parse_vector(data))

            print("Cells processed: ",global_counter,"/",len(self.paths_single_vector)-1)
            pd.concat(df_vector).to_csv('./dataStore/ExpDesign2_vector_.csv', index=False)  

        df_vector = pd.concat(df_vector)


    def parse_vector(self, data):

        df_all = []

        for count, cluster in enumerate(data.groupby(["ID_running"])):

            single_tags = cluster[0]
            single_tags = int(single_tags[0])

            single_data = cluster[1]
            single_data = single_data.reset_index(drop=True)

            single_data["rel_angle"] = 0
            
            spheroid_df = self.parse_cnt_dict(single_tags)

            if spheroid_df.shape[0] == 0:
                print("No data recorded from spheroid spheroid with running ID", single_tags)
                continue


            spheroid_df["dx"] = 0; spheroid_df["dy"] = 0; single_data["angle"] = 0
            spheroid_df.loc[1:,"dx"] = filter_Gauss(np.diff(spheroid_df["x"]))
            spheroid_df.loc[1:, "dy"] = filter_Gauss(np.diff(spheroid_df["y"]))

            for k in range(single_data.shape[0]):

                time = int(single_data["time"].values[k])
                row = spheroid_df[spheroid_df["time"] == time]

                if row.shape[0] == 0:
                    print("Time", time, "does not exist in spheroid frame cannot match protrusion, moving on")
                    continue

                v1 = np.stack((row["dx"].values[0], row["dy"].values[0]), axis = -1)
                v2 = np.stack((single_data["x_vec"].values[k], single_data["y_vec"].values[k]), axis = -1)
                angles_ = angle(v1, v2)

                deg_angles = np.rad2deg(angles_)
                #deg_angles = deg_angles[~np.isnan(deg_angles)]
                single_data.loc[k, "rel_angle"] = deg_angles

            df_all.append(single_data)

        df = pd.concat(df_all)
        #print(df, spheroid_df)

        return df


if __name__ == "__main__":
    host = Collect_Data()
    host.worker()
    #host.save()