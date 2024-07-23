import math as m
import numpy as np
import os
import pandas as pd

def vector_2d_length(u):
    return m.sqrt( (u[0]-u[2]) ** 2 + (u[1]-u[3]) ** 2)

def vector_2d_width(v):
    return m.sqrt((v[0]-v[2]) ** 2 + (v[1]-v[3]) ** 2)

def vector_angle(w):
    return np.arctan2((w[1]-w[3]), (w[0]-w[2]))

class Protrusion():

    def __init__(self, path, ctr):

        self.path = path
        self.ctr = ctr

        self.data_dict = {"path": [], "cell_line": [],"incubation_time": [], "x_1":[],"y_1":[],"z":[],"x_2":[],"y_2":[],"x_width3":[],"y_width3":[],"x_width4":[],"y_width4":[],"protrusion_length":[],"protrusion_width":[],"protrusion_direction":[] }


    def update_protrusion(self,cell_line,id,t,x1,y1,z,x2,y2,x3,y3,x4,y4):
        self.data_dict["path"].append(self.path_frame)
        self.data_dict["cell_line"].append(cell_line)
        self.data_dict["ID"].append(id)
        self.data_dict["incubation_time"].append(t)
        self.data_dict["x_1"].append(x1)
        self.data_dict["y_1"].append(y1)
        self.data_dict["z"].append(z)
        self.data_dict["x_2"].append(x2)
        self.data_dict["y_2"].append(y2)
        self.data_dict["x_width3"].append(x3)
        self.data_dict["y_width3"].append(y3)
        self.data_dict["x_width4"].append(x4)
        self.data_dict["y_width4"].append(y4)
        self.data_dict["protrusion_length"].append(vector_2d_length(x1,y1,x2,y2))
        self.data_dict["protrusion_width"].append(vector_2d_length(x3,y3,x4,y4))
        self.data_dict["protrusion_direction"].append(vector_angle(x1,y1,x2,y2))        

        print(self.data_dict)

  
    def save_data(self, id, ctr):
        print(ctr)

        if ctr["track"]:
            print("saved protrusion")
            df = pd.DataFrame.from_dict(self.data_dict)
            df.to_csv(os.path.join(self.path, "{}_{}_protrusion.csv".format(self.path, id)), index = False)

        #resetting dictionary after saving
        self.data_dict = {"path": [], "cell_line": [],"incubation_time": [], "protrusion_length":[],"protrusion_width":[],"protrusion_direction":[] }
        
    
