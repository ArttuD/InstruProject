# %%
from datetime import time
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import glob
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

#%%
paths=[]
paths = glob.glob(os.path.join('C:/Users/srboval1/IPN15/collagen/*','**','summary_ref_level.csv'))
print(paths)
concentration = '0/2'
type ='collagen'

a = []
for i in paths:
    path1 = os.path.split(os.path.split(i)[0])[0]
    path1.replace("/","\\")
    print(path1)
    splitted= path1.split('\\')[-1]
    tmp = pd.read_csv(i)
    tmp['coating_type'] = splitted.split('_')[-1]
    tmp['size'] = splitted.split('_')[-2]
    tmp['day'] = splitted.split('_')[-4]
    a.append(tmp)
    print(tmp)

a_concantenated=[]
a_concantenated = pd.concat(a)
a_concantenated['radius_(m)'] *= 1e6
a_concantenated = a_concantenated.rename(columns={'radius_(m)':'radius_(um)'})
a_concantenated['concentration'] = concentration
a_concantenated['type'] = type
a_concantenated['frequency'] = 0.05
a_concantenated = a_concantenated.reindex(columns=['day','frequency','concentration','type','sample','holder','location','repeat','track_id','reference_id','distance(um)','Cov_Sum','a_(um)','phi_(rad)','c','d','G_abs','radius_(m)','r2','rmse','inv.rmse','shift_(s)','a_error','phi_error','c_error','d_error','x','y','phi_(deg)','tan_phi'])

#%%
if os.path.exists('C:/Users/srboval1/IPN15/IPN15.csv'):
    a_new = pd.DataFrame(a_concantenated)
    a_new.to_csv('C:/Users/srboval1/IPN15/IPN15.csv', mode='a', index=False, header=False)
else:
    a_concantenated.to_csv("C:/Users/srboval1/IPN15/IPN15.csv", index=False)
    

