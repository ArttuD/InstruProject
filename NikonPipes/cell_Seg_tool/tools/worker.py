import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

class Worker():

    def __init__(self, path, ctr):

        self.path = path
        self.ctr = ctr

        self.path_frame = None

        self.mode = None

        self.data_dict_vector = {"path": [], "ID": [], "loc": [], "cell_id": [], "time": [], "index": [], "x": [], "y": [], "z": [], "x_vec": [], "y_vec": [], "angle": []}
        self.data_dict_dot = { "path": [], "ID": [], "loc": [], "cell_id": [],"time": [], "index": [], "x": [], "y": [], "z": []}

        self.current_idx = 0

    def update_points(self, x, y, id, index, t, z, v):

        self.data_dict_dot["path"].append(self.path_frame)
        self.data_dict_dot["ID"].append(id)
        self.data_dict_dot["loc"].append(v)
        self.data_dict_dot["cell_id"].append(id)
        self.data_dict_dot["time"].append(t)
        self.data_dict_dot["index"].append(index)
        self.data_dict_dot["x"].append(x)
        self.data_dict_dot["y"].append(y)
        self.data_dict_dot["z"].append(z)

        print(self.data_dict_dot)
              
    def update_vector(self, coords, id, index, t, z, v):
 
        x1 = coords[0][0]
        y1 = coords[0][1]

        x2 = coords[1][0]
        y2 = coords[1][1]

        x_vec = x2 - x1
        y_vec = y2 - y1

        self.data_dict_vector["path"].append(self.path_frame)              
        self.data_dict_vector["ID"].append(id)
        self.data_dict_vector["loc"].append(v)
        self.data_dict_vector["cell_id"].append(id)
        self.data_dict_vector["time"].append(t)
        self.data_dict_vector["index"].append(index)
        self.data_dict_vector["x"].append(x1)
        self.data_dict_vector["y"].append(y1)
        self.data_dict_vector["z"].append(z)
        
        self.data_dict_vector["x_vec"].append(x_vec)
        self.data_dict_vector["y_vec"].append(y_vec)
        self.data_dict_vector["angle"].append(np.arctan2(y_vec, x_vec))

    def save_data(self, id, ctr):
        print(ctr)

        if ctr["track"]:
            print("saved track")
            df = pd.DataFrame.from_dict(self.data_dict_dot)
            df.to_csv(os.path.join(self.path, "{}_{}_track.csv".format(self.path, id)), index = False)
        
        if ctr["vector"]:
            print("save vector")
            df = pd.DataFrame.from_dict(self.data_dict_vector)
            df.to_csv(os.path.join(self.path, "{}_{}_vector.csv".format(self.path, id)), index = False)


        self.data_dict_vector = {"path": [],"ID": [], "loc": [], "cell_id": [], "time": [], "index": [], "x": [], "y": [], "z": [], "x_vec": [], "y_vec": [], "angle": []}
        self.data_dict_dot = {"path": [],"ID": [], "loc": [], "cell_id": [],"time": [], "index": [], "x": [], "y": [], "z": []}
        
        self.current_idx = 0
    
