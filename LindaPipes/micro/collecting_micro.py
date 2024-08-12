#%%
from datetime import time
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from glob import glob
from seaborn import palettes
from seaborn.categorical import swarmplot
from tqdm import tqdm
import json
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter1d
import os
import re
import math
import matplotlib.ticker as tck
import argparse

#Directory with all micro data
paths = glob(os.path.join('C:/Users/srboval1/OneDrive - Aalto University/micromanipulator/IPN_LVLG_15mgml/*/*/results/summary_ID_level.csv'))
print(paths)

a = []
names2 = []

for i in paths:
    parts3 = os.path.split(os.path.split(os.path.split(i)[0])[0]) #3rd level contains all info needed to be added

    concentration =  (os.path.split(parts3[1])[1]).split('_')[1]
    print('concetration'+concentration)
    bead_size =  (os.path.split(parts3[1])[1]).split('_')[2]
    print('size'+bead_size)
    bead_coating =  (os.path.split(parts3[1])[1]).split('_')[3]
    print('coating'+bead_coating)

    tmp = pd.read_csv(i)

    tmp['bead_size'] = bead_size
    tmp['coating_type'] = bead_coating
    tmp['crosslinker'] = concentration
    a.append(tmp)

a = pd.concat(a)


a.to_csv("C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/micro/all_micro.csv", index=False)


# %%
