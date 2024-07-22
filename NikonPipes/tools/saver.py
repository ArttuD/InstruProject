import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math as m

def vector_2d_length(v):
    return m.sqrt((v[0]-v[2]) ** 2 + (v[1]-v[3]) ** 2)

class Cells():

    def __init__(self, path):

        self.path = path
        self.reset_frame()

    def reset_frame(self):
        self.data_dict =  {"path": [], "cell_id": [], "time": [], "location": [], "x": [], "y": [], "z": [], "x_2": [], "y_2": [], "z_2": [], "lenght":[], "x_vec":[], "y_vec":[], "angle":[], "width":[] }

    def update_cell(self, det_id, t, x1, y1, z1, loc):
        
        self.data_dict["cell_id"].append(det_id)
        self.data_dict["time"].append(t)
        self.data_dict["location"].append(loc)
        self.data_dict["x"].append(x1)
        self.data_dict["y"].append(y1)
        self.data_dict["z"].append(z1)

        self.save_data("cell")
              
    def update_vector(self, coords, id, t, z, v, coords_):
 
        x1 = coords[0][0]
        y1 = coords[0][1]

        x2 = coords[1][0]
        y2 = coords[1][1]

        x_vec = x2 - x1
        y_vec = y2 - y1

        self.data_dict["path"].append(self.path_frame)              
        self.data_dict["ID"].append(id)
        self.data_dict["loc"].append(v)
        self.data_dict["cell_id"].append(id)
        self.data_dict["time"].append(t)
        self.data_dict["x"].append(x1)
        self.data_dict["y"].append(y1)
        self.data_dict["z"].append(z)
        
        self.data_dict["x_vec"].append(x_vec)
        self.data_dict["y_vec"].append(y_vec)
        self.data_dict["legth"].append(vector_2d_length(x1,y1,x2,y2))
        self.data_dict["angle"].append(np.arctan2(y_vec, x_vec))

        x1 = coords_[0][0]
        y1 = coords_[0][1]

        x2 = coords_[1][0]
        y2 = coords_[1][1]

        self.data_dict_dot["width"].append(vector_2d_length(x1,y1,x2,y2))

        self.save_data("vector")

    def save_data(self, ctr):

        if ctr == "cell":
            print("saved track")
            df = pd.DataFrame.from_dict(self.data_dict_dot)
            df.to_csv(os.path.join(self.path, "{}_track.csv".format(self.path)), index = False)
        
        if ctr == "vector":
            print("save vector")
            df = pd.DataFrame.from_dict(self.data_dict_vector)
            df.to_csv(os.path.join(self.path, "{}_vector.csv".format(self.path)), index = False)
    
