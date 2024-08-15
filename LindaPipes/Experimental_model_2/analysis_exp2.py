#%%
from PIL.ExifTags import TAGS
import pathlib
import csv
import glob
from PIL import Image
import pandas as pd
import os
from datetime import time
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns
import glob
from seaborn import palettes
from seaborn.categorical import swarmplot
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import json
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter1d
from skimage.color import rgb2gray
import os
import re
import math
import matplotlib.ticker as tck
import argparse
import cv2
import skimage as ski

#%%
path = 'C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles'
Exp2 = pd.read_csv(path +'/contours_exp2.csv', index_col=None)


#%%
unique_materials = Exp2['material'].unique()
print(unique_materials)
unique_materials = np.delete(unique_materials,2)
print(unique_materials)
unique_lines = Exp2['cell_line'].unique()
print(unique_lines)

# %%
color=['red','blue','green']


palette = {
    'MCF10A': 'tab:red',
    'MCF10A.DCIS.COM': 'tab:blue',
    'MCF10AT': 'tab:green',
}

lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-') for c in color]
labels = ['MCF10A', 'MCF10A.DCIS.COM', 'MCF10AT']
# %%
fig, axs = plt.subplots(1, len(unique_materials), figsize=(17,5))
plt.subplots_adjust(wspace=0.25, hspace=0.25)
fig.suptitle("Time-series of spheroid areas", fontsize=18, y=1)

for i,unique_material in enumerate(unique_materials):
    specific_df = df_plot[df_plot['material'] == unique_material]
    specific_df_grp = specific_df.groupby(['cell_line','incubation_time']).mean()
    #print(specific_df_grp.head())
    ax=axs[i]
    ax.plot(specific_df_grp.unstack(0)['area'],color=color) # area as a function of incubation time, for each cell line
    ax.set_xlabel('Incubation Time [h]')  
    ax.set_ylabel('Mean Area')        
    ax.set_title(unique_material)


# %%
fig, axs = plt.subplots(3, len(unique_materials), figsize=(30,10))
plt.subplots_adjust(wspace=0.25, hspace=0.6)
fig.suptitle("Area time-series", fontsize=20, y=0.95)


for i, unique_material in enumerate(unique_materials):
    specific_df = Exp2[(Exp2['material'] == unique_material) & (Exp2['area'] > 0) ]

    min_value_all = (specific_df['incubation_time'].min() // 24) * 24 #latest end of timelpases x_max
    max_value_all = (specific_df['incubation_time'].max() // 24 + 2) * 24 #latest end of timelpases x_max

    times = list(np.arange(min_value_all, max_value_all, 24)) #ticks' values

    
    for j,unique_line in enumerate(unique_lines):
        ax=axs[j,i]
        specific_specific_df = specific_df[(specific_df['cell_line'] == unique_line)]

        #min_value = specific_specific_df['incubation_time'].min() #latest end of timelpases x_max
        #max_value = specific_specific_df['incubation_time'].max() #latest end of timelpases x_max

        #print( f'{unique_material}- {unique_line}; min: {min_value} and max {max_value}')

        sns.swarmplot(specific_specific_df, y='area', ax=ax, x='incubation_time', native_scale=True, hue='spheroid_id', size =3, marker='o')
        ax.set_xlim(min_value_all, max_value_all)  

        ax.set_yscale('log')

        ax.set_ylabel('Area [µm2]', fontsize= 16)  
        ax.set_xlabel('Incubation time [hour]', fontsize= 16)  

        ax.set_title(f'{unique_material} - {unique_line}', fontsize= 17)
        ax.set_ylim(0.001,0.5)
        ax.set_xticks(times)
        ax.set_xticklabels(times)
        ax.tick_params(axis='both', which='major', labelsize=15)  # Major ticks

    
plt.savefig(path +'/Time_series.png',dpi=1000)


# %%
