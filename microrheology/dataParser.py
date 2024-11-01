#%%
import glob
import pandas as pd
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt
#%%

paths = glob.glob("H:/instru_projects/rheology/microrheology/paper_*/Alg*/*/results/*_ID_*.csv")
 
for i in range(len(paths)):

    current = paths[i]
    radius_data = None
    radius_file = os.path.join(os.path.split(current)[0],'radius_estimates.json')

    if os.path.exists(radius_file):
        with open(radius_file,'r') as f:
            radius_data = json.load(f)

    df_tmp = pd.read_csv(current)
    parts = current.split("\\")[2].split("_")
    parts_2 = current.split("\\")[3].split("_")
    parts_3 = current.split("\\")[1].split("_")
   
    df_tmp["material"] = parts_3[1]
    df_tmp["day"] = parts_2[0]
    df_tmp["size"] = int(parts_2[1][:-2])
    df_tmp["coating"] = parts_2[2]
    df_tmp["type"] = current.split("\\")[2]

    if i == 0:
        df_micro = df_tmp
    else:
        df_micro = pd.concat((df_micro, df_tmp))
 
df_micro.to_csv("./dataStore/datas_2.csv")


#%%

for tag, data in df_micro.groupby(["type"]):
    fig, axes = plt.subplots(1, 1, figsize=(15, 8), dpi=200)
    print(tag)
    sns.swarmplot(x = "radius_(m)", y= "G_abs", data = data, hue= "day", ax = axes)
    plt.setp(axes.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()
# %%

df_micro.keys()