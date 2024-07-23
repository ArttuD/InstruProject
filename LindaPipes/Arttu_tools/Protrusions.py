import math as m
import numpy as np
import os
import pandas as pd

def vector_2d_length(v):
    return m.sqrt( ** 2 + (v[1]-v[3]) ** 2)

def vector_angle(w):
    return np.arctan2((w[1]-w[3]), (w[0]-w[2]))

class Protrusion():

    def __init__(self, path, ctr):

        self.path = path
        self.ctr = ctr

        self.data_dict = {"path": [], "cell_line": [], "cell_id": [], "time": [], "x_1": [], "y_1": [], "z_1": [], "x_2": [], "y_2": [], "z_1": [], "lenght":[] }

        self.current_idx = 0

    def update_protrusion(self,x1,y1,x2,y2,t,z, cell_line, cell_id):
        self.data_dict["path"].append(self.path_frame)
        self.data_dict["cell_line"].append(cell_line)
        self.data_dict["cell_id"].append(cell_id)
        self.data_dict["time"].append(t)
        self.data_dict["x1"].append(x1)
        self.data_dict["y1"].append(y1)
        self.data_dict["z1"].append(z)
        self.data_dict["x2"].append(x2)
        self.data_dict["y2"].append(y2)
        self.data_dict["z2"].append(z)
        self.data_dict["lenght"].append(vector_2d_length(x1,y1,x2,y2))
        self.data_dict["angle"].append(vector_angle(x1,y1,x2,y2))        

        print(self.data_dict)

  
    def save_data(self, id, ctr):
        print(ctr)

        if ctr["track"]:
            print("saved track")
            df = pd.DataFrame.from_dict(self.data_dict)
            df.to_csv(os.path.join(self.path, "{}_{}_protrusion.csv".format(self.path, id)), index = False)


        self.data_dict = {"path": [], "cell_line": [], "cell_id": [], "time": [], "x_1": [], "y_1": [], "z_1": [], "x_2": [], "y_2": [], "z_1": [], "lenght":[] }
        
        self.current_idx = 0
    
