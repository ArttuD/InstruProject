import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

class Worker():

    def __init__(self, ctr, path):

        self.ctr = ctr
        self.path = path

        self.data_dict_vector = None
        self.data_dict_dot = None

        self.current_idx = 0

    def update_points(self):

        self.data_dict_dot = { "ID": [], "loc": [], "cell_id": [],"time": [], "index": [], "x": [], "y": [], "z": [], "x_mag": [], "y_mag": []}
        for i in self.data_dict_dot.keys():
            if i == "index":
                self.data_dict_dot[i] = self.current_idx 
                self.current_idx += 1
            else:
                self.data_dict_dot[i] = self.ctr[i]

    def update_vector(self):

        self.data_dict_vector = {"ID": [], "loc": [], "cell_id": [], "time": [], "index": [], "x": [], "y": [], "z": [], "x_mag": [], "y_mag": [], "angle": []}

        for i in self.data_dict_vector.keys():
            if i == "index":
                self.data_dict_vector[i] = self.current_idx 
                self.current_idx += 1
            else:
                self.data_dict_vector[i] = self.ctr[i]

    def save_data(self):
        
        df = pd.from_dict(self.data_dict)
        df.to_csv(os.path.join(self.path, "{}_{}_{}_tooled.csv".format(self.ctr["ID"], self.ctr["loc"])))

        self.data_dict_vector = None
        self.data_dict_dot = None

        self.current_idx = 0

        self.ctr["close"] == False

    
