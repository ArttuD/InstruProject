
# %%

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 
import glob

import pandas as pd
import seaborn as sns
import csv

import json


print("updating ignroe and doubles")

path_meta = "./dataStore/metalib.json"
with open(path_meta, "r") as f:
    own_meta = json.load(f)

path = "./dataStore/ignore.json"
with open(path, "r") as f:
    datas = json.load(f)

for count, ckey in enumerate(datas.keys()):

    own_meta[ckey]["ignore"] = (np.array(datas[ckey]["ignore"])-1).tolist()
    own_meta[ckey]["multi"] = (np.array(datas[ckey]["multi"])-1).tolist()


with open('./dataStore/metalib.json', 'w', encoding='utf-8') as f:
    json.dump(own_meta, f, ensure_ascii=False, indent=4)


"""
##240306 -> 240303, Write overgrown and label it as overgrown

# %%

path_meta = "./dataStore/metalib.json"
with open(path_meta, "r") as f:
    own_meta = json.load(f)

path = "./dataStore/ignore.json"
with open(path, "r") as f:
    datas = json.load(f)

for count, ckey in enumerate(datas.keys()):

    own_meta[ckey]["ignore"] = (np.array(datas[ckey]["ignore"])-1).tolist()
    own_meta[ckey]["multi"] = (np.array(datas[ckey]["multi"])-1).tolist()

own_meta[ckey].keys()

for ckey in own_meta.keys():
    if "ignore" in own_meta[ckey].keys():
        pass
    else:
        print("missing from", ckey)

# %% 

for ckey in own_meta.keys():
    if (own_meta[ckey]["matrix"] == "collagen"):
        for count, cLabels in enumerate(own_meta[ckey]["cell"]):
            if cLabels in ["T", "MCF10AT"]:
                own_meta[ckey]["ignore"].extend([count])


# %%

with open('./dataStore/metalib.json', 'w', encoding='utf-8') as f:
    json.dump(own_meta, f, ensure_ascii=False, indent=4)

# %%
"""