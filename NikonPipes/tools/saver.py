import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from glob import glob

def vector_2d_length(v):
    return m.sqrt((v[0]-v[2]) ** 2 + (v[1]-v[3]) ** 2)

class Cells():

    def __init__(self, path):

        self.path = path
        self.prot_table = None
        self.reset_frame()
        self.search_old()

    def search_old(self):
        vector = glob(os.path.join( self.path,"*_vector.csv"))
        cells = glob(os.path.join( self.path,"*_vector.csv"))
        print(vector, cells)
        if len(vector) > 0:
            self.data_dict = pd.read_csv(vector[0]).to_dict('series')
        if len(cells) > 0: 
            self.data_dict_ = pd.read_csv(cells[0]).to_dict('series')

    def reset_frame(self):
        self.data_dict =  {"cell_id": [], "time": [], "location": [], "x": [], "y": [], "z": [],"x2": [], "y2": [], "z2": [], "lenght":[], "x_vec":[], "y_vec":[], "angle":[], "width":[] }
        self.data_dict_ =  {"cell_id": [], "time": [], "location": [], "x": [], "y": [], "z": []}

    def return_timestep(self,location, time):

        df = pd.DataFrame.from_dict(self.data_dict)
        self.prot_table = df
        
        return self.prot_table[(self.prot_table["location"] == location) & (self.prot_table["time"] == time)].reset_index(drop=True)

    def update_cell(self, det_id, t, x1, y1, z1, loc):
        
        self.data_dict_["cell_id"].append(det_id)
        self.data_dict_["time"].append(t)
        self.data_dict_["location"].append(loc)
        self.data_dict_["x"].append(x1)
        self.data_dict_["y"].append(y1)
        self.data_dict_["z"].append(z1)

        self.save_data("cell")
              
    def update_vector(self, coords, id, t, z, v, coords_):
 
        x1 = coords[0][0]
        y1 = coords[0][1]

        x2 = coords[1][0]
        y2 = coords[1][1]

        x_vec = x2 - x1
        y_vec = y2 - y1
  
        self.data_dict["location"].append(v)
        self.data_dict["cell_id"].append(id)
        self.data_dict["time"].append(t)
        self.data_dict["x"].append(x1)
        self.data_dict["y"].append(y1)
        self.data_dict["z"].append(z)
        self.data_dict["x2"].append(x2)
        self.data_dict["y2"].append(y2)
        self.data_dict["z2"].append(z)
        
        self.data_dict["x_vec"].append(x_vec)
        self.data_dict["y_vec"].append(y_vec)
        self.data_dict["lenght"].append(vector_2d_length([x1,y1,x2,y2]))
        self.data_dict["angle"].append(np.arctan2(y_vec, x_vec))

        x1 = coords_[0][0]
        y1 = coords_[0][1]

        x2 = coords_[1][0]
        y2 = coords_[1][1]

        self.data_dict["width"].append(vector_2d_length([x1,y1,x2,y2]))

        self.save_data("vector")

    
    def save_data(self, ctr):

        if ctr == "cell":
            print("saved track")
            df = pd.DataFrame.from_dict(self.data_dict_)
            df.to_csv(os.path.join(self.path, "data_track.csv"), index = False)
        
        if ctr == "vector":
            print("save vector")
            df = pd.DataFrame.from_dict(self.data_dict)
            self.prot_table = df
            df.to_csv(os.path.join(self.path, "data_vector.csv"), index = False)

    
