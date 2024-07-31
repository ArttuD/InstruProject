#%%
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# %%
#Deserializing dictionary saved in pickle
with open('C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp_design_2_ID.pkl', 'rb') as f:
    total_dict_ = pickle.load(f)

#%%
#Keys of nested dictionaries
print(total_dict_.keys())
#Keys-indexes nested in the first internal dictionary
print(total_dict_[0].keys())
print(total_dict_[115]["cell_line"])
#%%
for i in range(len(total_dict_[1]["masks"])):
    print(i)
#%%
total_dict_[1]["time"][1]
#%%                   
len(total_dict_[1]["masks"])
#%%
concentration_mapping = {
'2mgml': '1',
'3mM': '2',
'40mM': '3'
}
material_mapping = {
'2mgml': '1',
'3mM': '2',
'40mM': '3'
}
reverse_material_mapping = {
'2mgml':'collagen 2mg/ml',
'3mM':'IPN 2.5mM',
'40mM':'IPN 40mM'
}
cell_mapping = {
'MCF10A': '1',
'DCIS.COM': '2',
'MCF10A.DCIS.COM': '2',
'MCF10AT': '3'
}

#%%
pixel_size = 0.32*1e-6
df_collecting = []
df_all = []

for key in total_dict_.keys() :
    df_temp =[] 

    for i in range(len(total_dict_[key]["time"])): #looping through time
        time = total_dict_[key]["time"][i]
        mask = total_dict_[key]["masks"][i]
        cell_line = total_dict_[key]["cell_line"]
        matrix = total_dict_[key]["matrix"]
        area = cv2.contourArea(mask)                
        area_µm = area*pixel_size    
        cell_label = cell_line
        well_id = 0
        measurement_id = key
        matrix = matrix
        perimeter = cv2.arcLength(mask, closed=True)*pixel_size #perimeter for each contour
        compactness = 4 * math.pi * area_µm  #compactness (spheroid area/area of object with the identical perimeter)
        rect = cv2.minAreaRect(mask) #minimal rectangle
        width, height  = rect[1] #extracting the tuple containing rectangular dimensions
        aspect_ratio = width / height
        convexhull = cv2.convexHull(mask)
        convexhull_perimeter = cv2.arcLength(convexhull, closed=True)*pixel_size #perimeter for each contour
        convexhull_area = cv2.contourArea(convexhull)*pixel_size
        convexity_perimeter = convexhull_perimeter/perimeter
        convexity_area = convexhull_area/area
        
        info={}
        info = {
            "spheroid_id": key+1,
            "time_id": i+1,
            "date": 0,
            "cell_line":cell_label,
            "cell_line_no": int(cell_mapping.get(cell_label)),
            "incubation_time": time,
            "material": matrix,
            "concentration": 0,
            "area": area_µm,
            "perimeter": perimeter,
            "convex hull area": convexhull_area,
            "convex hull perimeter": convexhull_perimeter,
            "convexity_perimeter": convexity_perimeter,
            "convexity_area": convexity_area,
            "compactness": compactness,
            "aspect_ratio": aspect_ratio
            }
        df_temp.append(pd.DataFrame([info]) )  #appending the info to the contours_info, which collects all timestemps
    
    df_collecting.append(pd.concat(df_temp,ignore_index=True)) #lists of dfs are contantenated and appended to df_collecting, thedf_temp is re-initialized

df_all=pd.concat(df_collecting, ignore_index=True)
#%%
print(df_all.shape)

#%%
print(df_all)
# %%
saving_path= 'C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles'
df_all.to_csv(saving_path+'/contours_exp2.csv')  
# %%
