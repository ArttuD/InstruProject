import math as m
import numpy as np
import os
import pandas as pd


class Protrusion():

    def __init__(self, path, ctr):

        self.path = path
        self.ctr = ctr

        self.data_dict = {"path": [], "cell_line": [], "cell_line": [],"incubation_time": [], "x":[],"y":[],"z":[] }


    def update_protrusion(self,cell_line,id,t,x,y,z):
        self.data_dict["path"].append(self.path_frame)
        self.data_dict["cell_line"].append(cell_line)
        self.data_dict["ID"].append(id)
        self.data_dict["incubation_time"].append(t)
        self.data_dict["x"].append(x)
        self.data_dict["y"].append(y)
        self.data_dict["z"].append(z)
        print(self.data_dict)

  
    def save_data(self, id, ctr):
        print(ctr)

        if ctr["track"]:
            print("saved cell")
            df = pd.DataFrame.from_dict(self.data_dict)
            df.to_csv(os.path.join(self.path, "{}_{}_cell.csv".format(self.path, id)), index = False)

        #resetting dictionary after saving
        self.data_dict = {"path": [], "cell_line": [],"incubation_time": [], "x_1":[],"y_1":[],"z":[],"x_2":[],"y_2":[],"x_width3":[],"y_width3":[],"x_width4":[],"y_width4":[],"protrusion_length":[],"protrusion_width":[],"protrusion_direction":[] }
        
    
