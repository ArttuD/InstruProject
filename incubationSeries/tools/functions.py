import os
import csv

import cv2
import skimage
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from sklearn.preprocessing import RobustScaler
import scipy
import json

from sklearn.decomposition import PCA

def saver(root, fig_path, area, perimeter, fb, cutThreshold, indexCell, fig, mask, indexImg):
    result_main = os.path.join(root,"results")
    mag = fig_path.split(os.sep)[-1].split("_")[0]

    number = fig_path.split("/")[-1].split(os.sep)[-1].split("_")[1][:-4]

    if not os.path.exists(result_main):
        os.makedirs(result_main)

    parts = os.path.split(fig_path)[0].split(os.sep)
    cell_line = parts[1]
    condition = parts[2]
    time = parts[3]
    save_folder = os.path.join(result_main,os.path.join(cell_line,condition,time))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    path_csv = os.path.join(result_main,"results.csv")

    if not os.path.exists(path_csv):
        with open(path_csv,mode = 'w', newline = "") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["name", "area(pixel)", "perimeter(pixel)", "x_1", "y_1", "x_2", "y_2", "figPath",  "cutThreshold", "index_cell"])

    name = os.path.join(save_folder,"{}_contour_img_{}_cell_{}.npy".format(mag,number, indexCell))
    name_figure = os.path.join(save_folder,"{}_contour_img_{}_cell_{}.png".format(mag,number, indexCell))

    with open(path_csv,mode = 'a', newline = "") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([name, area, perimeter, int(fb[0][0][0]),int(fb[0][0][1]),int(fb[0][1][0]),int(fb[0][1][1]), fig_path, cutThreshold, indexCell]) 

    np.save(name, mask)
    fig.savefig(name_figure)
    

def increase_brightness(img, value=1):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def faridFilter(img, filterCoef):

    def mousePoints(event,x,y,flags,param):
        #Crop image
        global refPt
        global image
        # Left button click
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            final_boundaries.append((refPt[0],refPt[1]))
            #cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 25)
            cv2.imshow("win", frame)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            clone = frame.copy()
            cv2.rectangle(clone, refPt[0], (x, y), (0, 255, 0), 4)
            cv2.imshow("win", clone)

    fig, ax = plt.subplots(1, 2,figsize=(10, 5))

    #img = increase_brightness(cv2.imread(figs[indexImg]))
    frame = img.copy()

    #global final_boundaries
    final_boundaries = []

    cv2.namedWindow('win',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("win", mousePoints)

    cv2.imshow("win",frame)

    k = cv2.waitKey(0)       
    # Destroying present windows on screen
    cv2.destroyAllWindows()

    crop = frame[final_boundaries[0][0][1]:final_boundaries[0][1][1],final_boundaries[0][0][0]:final_boundaries[0][1][0],:]

    yabs = int(np.abs(final_boundaries[0][0][1]-final_boundaries[0][1][1])/2)
    xabs = int(np.abs(final_boundaries[0][0][0]-final_boundaries[0][1][0])/2)

    ax[0].set_title("Cropped image")
    ax[0].imshow(crop)

    cropBW = skimage.color.rgb2gray(crop)

    g = skimage.filters.gaussian(cropBW,filterCoef)
    #g = skimage.filters.butterworth((g), order = 1,high_pass = False)

    edges = skimage.filters.farid(g)

    ax[1].set_title("after filter")

    im1 = ax[1].imshow(1-edges, cmap = "inferno")
    plt.colorbar(mappable = im1)

    plt.show()

    return edges, final_boundaries

def segmentImg(img, edges, final_boundaries, cutThreshold, idx):
    
    ClosingCoef = 15
    whiteTopCoef = 5
    dilationCoedf = 5

    frame = img.copy()
    
    e = 1-edges<cutThreshold

    e = skimage.morphology.closing(e, skimage.morphology.disk(ClosingCoef))
    e2 = skimage.morphology.white_tophat(e,skimage.morphology.disk(whiteTopCoef))
    e2 = skimage.morphology.dilation(e2,skimage.morphology.disk(dilationCoedf))

    tmp = e.astype("int")-e2.astype("int")
    tmp[tmp<0] = 0
    tmp = tmp.astype("uint8")

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)                          

    # draw contours on the original image
    image_copy = np.zeros_like(frame[:,:,0].copy())
    areas = [cv2.contourArea(c) for c in contours]

    largest_idx = np.argmin(np.abs(np.array(sorted(areas)[idx])-areas))
    largest = contours[largest_idx]
    
    perimeter =  cv2.arcLength(largest,True)

    area = cv2.contourArea(largest)

    for i in range(len(largest[:])):
        largest[i][0][0] += final_boundaries[0][0][0] #xabs
        largest[i][0][1] += final_boundaries[0][0][1] #yabs

    cv2.fillPoly(image_copy, pts = [largest], color=(255,255,255))

    col = np.zeros_like(frame[:,:,0])
    col[image_copy.astype("bool")] = 255.
    mask = np.zeros_like(frame)
    mask[col.astype("bool"),2] = 255.
    out = cv2.addWeighted(frame,0.95,mask,10,0.)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=col, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=frame, contours=[largest], contourIdx=-1, color=255, thickness=3)


    fig, ax = plt.subplots(2, 2,figsize=(10, 7), facecolor='white')

    ax[0,0].imshow(frame)
    ax[0,1].imshow(mask)
    ax[1,0].imshow(out)
    ax[1,1].imshow(frame[final_boundaries[0][0][1]:final_boundaries[0][1][1],final_boundaries[0][0][0]:final_boundaries[0][1][0],:])

    return fig, area, perimeter, largest, mask

def loadJson(path):
    #print(path)
    f = open(path)
    data = json.loads(f.read())
    #print(data)
    dataRep = []
    
    for j in range(len(data["shapes"])):
        img  = np.zeros((1544,2064))
        cv2.fillPoly(img, pts = [np.array(data["shapes"][j]["points"]).astype("int")], color=(255,255,255))
        frame = img.copy()
        col = np.zeros_like(frame)
        col[img.astype("bool")] = 255.
        mask = np.zeros_like(frame)
        mask[col.astype("bool")] = 255.
        out = cv2.addWeighted(frame,0.95,mask,10,0.)

        contours, hierarchy = cv2.findContours(image=col.astype("uint8"), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        dataRep.append(contours)
        
    #print(len(dataRep), len(data["shapes"]))
    if len(data["shapes"]) > 1:
        for i in range(len(data["shapes"])):
            shapes = dataRep[i][0].shape
            if shapes[0] < 2:
                del dataRep[i]
        return dataRep, len(dataRep)
    else:
        return dataRep, len(dataRep)

def loadNpy(path): 
    #print(path)
    Imgcnt = np.load(path)
    contours, hierarchy = cv2.findContours(image=Imgcnt[:,:,2], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    return contours, 1

def findId(path):
    
    parts = path.split("\\")
    if len(parts[-1].split("_")[0]) > 2:
        mag = parts[-1].split("_")[0][:-1]
    else:
        mag = parts[-1].split("_")[0]

    if "10" in mag:
        m = 0.46
    elif "20" in mag:
        m = 0.23
    elif ("5" in mag) | ("4" in mag):
        m = 1.15
    else:
        print("cannot define pixel size for ", mag, "\n In path: ", path)
        m = 0

    cellLabel = parts[1]
    day = parts[0].split("/")[-2].split("_")[0] #parts[2]
    conc = parts[0].split("/")[-2].split("_")[1] #parts[3]
    condition = parts[2]
    if len(parts[3])>4:
        time = float(parts[3][:-5])
    else:
        time = float(parts[3][:-1])
    ending = parts[-1].split(".")[1]

    # condition = parts[2]
    # time = parts[3]
    
    return cellLabel, day, conc, time, m, ending, condition

def processContour(contours,j):
    data = contours[j]
    return cv2.contourArea(data), cv2.arcLength(data,True), data

def NormalizeData(data):
    return (data-data.min()) / (data.max()-data.min())
    #return data

def normalize(datas):
    return (datas-np.mean(datas))/np.std(datas)

def norm(data):
    trans = RobustScaler()
    return trans.fit_transform(data.reshape(-1, 1))[:,0], trans

def sample_freqs(data,j):

    xx = data[:,0,:]
    xx = (xx.astype(np.float32)-np.mean(xx,axis=0)[None,...])/np.std(xx,axis=0)[None,...]

    r = np.diff(scipy.ndimage.gaussian_filter(np.sqrt(xx[:,0]**2 + xx[:,1]**2),2))
    #phi = np.diff(scipy.signal.detrend(scipy.ndimage.gaussian_filter(np.unwrap(np.arctan2(xx[:,0],xx[:,1])),2)))
    phi = np.diff(scipy.ndimage.gaussian_filter(np.unwrap(np.arctan2(xx[:,0],xx[:,1])),2))
    xxPolar = np.stack((r,phi), axis = 1)

    t = np.arange(len(r))

    freq = np.fft.fftfreq(t.shape[-1])
    fft_xy = np.fft.fft(xxPolar, axis = 0)
    powerAll = np.abs(fft_xy)**2

    return fft_xy, powerAll, freq


def PCA_decomposition(data_dict):
    max_len = 0
    max_idx = 0

    data_dict["pw_tot"] = []

    for i in range(len(data_dict["cnt"])):
        current_len = len(data_dict["cnt"][i])
        if current_len > max_len:    
            max_len = current_len
            max_idx = i

    pw_pos = np.zeros(50, dtype="float64")
    pw_neg = np.zeros(50, dtype="float64")


    for i in range(len(data_dict["cnt"])):

        fft_xy, pw, freq = sample_freqs(data_dict["cnt"][i],i)
        
        pos_freq = np.where(freq>=0)[0]
        neg_freq = np.where(freq<0)[0][::-1]

        pw_pos = pw[pos_freq[:50],:]
        pw_neg = pw[neg_freq[:50],:]

        if (len(pw_pos) < 50) | (len(pw_neg) < 50):
            data_dict["pw_tot"].append(np.zeros((50,2)) + np.zeros((50,2)))
        else:
            data_dict["pw_tot"].append(pw_pos + pw_neg)

    return data_dict

def PCA_and_viz(dict, feature, hue_label, IDs):

    headerR = np.array(list(map(lambda x: "r_" + x,np.arange(50).astype("str"))))
    headerPhi = np.array(list(map(lambda x: "p_" + x, np.arange(50).astype("str"))))

    extra = np.array(['label', 'day', 'conc', 'time', 'runNum', 'expr', 'pixelSize'])
    tempDict = {}

    for i in range(50):
        if feature == "R":
            tempDict[headerR[i]] = []
        else:
            tempDict[headerPhi[i]] = []
        
    for i in range(len(dict["pw_tot"])):
        for j in range(50):
            if feature == "R":
                tempDict[headerR[j]].append(dict["pw_tot"][i][j,0])
            else:
                tempDict[headerPhi[j]].append(dict["pw_tot"][i][j,1])

    df_pca = pd.DataFrame()

    for i in extra:
        df_pca[i] = dict[i]

    for i in tempDict.keys():
        df_pca[i] = normalize(np.array(tempDict[i]))

    df_pca = df_pca.replace(np.nan, 0)

    fig, ax = plt.subplots(2, 3,figsize=(16, 10),facecolor='white')

    pca, components = performPCA(df_pca)

    comp = pd.DataFrame(components, columns = ['1','2', '3'])

    comp["label"] = df_pca["label"]
    comp["time"] = df_pca["time"]
    comp["pixelSize"] = df_pca["pixelSize"]

    total_var = pca.explained_variance_ratio_.sum() * 100

    ax[0,0].set_title("{} hours: explained {:.2f}%".format(i, total_var))
    ax[0,0].set_xlabel("1st component")
    ax[0,0].set_ylabel("2nd component")
    sns.scatterplot(x='1', y='2', data=comp, ax = ax[0,0], hue=hue_label)
    ax[0,0].legend(loc='upper right')

    ax[0,1].set_xlabel("2nd component")
    ax[0,1].set_ylabel("3rd component")
    sns.scatterplot(x='2', y='3', data=comp, ax = ax[0,1], hue=hue_label)
    ax[0,1].legend(loc='upper right')


    ax[0,2].set_xlabel("1st component")
    ax[0,2].set_ylabel("3rd component")
    sns.scatterplot(x='1', y='3', data=comp, ax = ax[0,2], hue=hue_label)
    ax[0,2].legend(loc='upper right')

    print('In {} hours, total Explained Variance: {}%'.format(i,total_var))

    weights = pca.components_

    ax[1,0].bar(np.arange(0,len(weights[0,::2])),weights[0,::2], label = "y-1", alpha = 0.5)
    #ax[1,0].bar(np.arange(0,len(weights[0,1::2])),weights[0,1::2], label = "y-1", alpha = 0.5)
    ax[1,0].legend()

    ax[1,1].bar(np.arange(0,len(weights[0,::2])),weights[1,::2], label = "y-2", alpha = 0.5)
    #ax[1,1].bar(np.arange(0,len(weights[0,1::2])),weights[1,1::2], label = "y-2", alpha = 0.5)
    ax[1,1].legend()

    ax[1,2].bar(np.arange(0,len(weights[0,::2])),weights[2,::2], label = "y-3", alpha = 0.5)
    #ax[1,2].bar(np.arange(0,len(weights[0,1::2])),weights[2,1::2], label = "y-3", alpha = 0.5)
    ax[1,2].legend()
    ax[1,2].set_xlabel("Frequency [Hz]")
    fig.savefig(os.path.join("./data", "PCA_{}_{}_{}".format(IDs[0],IDs[1],IDs[2])))
    return pca, df_pca, components


def pipe(datas, experiment_number):

    data_dict = {"path":[], "areas": [], "perimeter": [] ,"label":[],"day":[],
                 "conc":[],"cnt":[], "pixelSize": [], "time":[], "runNum":[], 
                 "expr":[], "ending": [], "condition": []} 
    
    for i in datas:
        #print(i)
        label, day, conc, time, m, ending, condition = findId(i)

        if ending == "npy":
            cnt, contour = loadNpy(i)
            typeFlag = 0
        elif ending == "json":
            cnt, contour = loadJson(i)
            typeFlag = 1       
        else:
            print("Unknown Ending in path ", i)
            typeFlag = 3
        #return cnt, contour
        #return cnt,contour
        if typeFlag != 3:
            for j in range(contour):
                if typeFlag == 0:
                    area, contReal, data = processContour(cnt,0)
                else:
                    area, contReal, data = processContour(cnt[j],0)
                
                data_dict["path"].append(i)
                data_dict["areas"].append(area)
                data_dict["perimeter"].append(contReal)
                data_dict["label"].append(label)
                data_dict["day"].append(day)
                data_dict["cnt"].append(data)
                data_dict["pixelSize"].append(m)
                data_dict["time"].append(time)
                data_dict["runNum"].append(j)
                data_dict["conc"].append(conc)
                data_dict["expr"].append(experiment_number)
                data_dict["ending"].append(ending)
                data_dict["condition"].append(condition)

        data_dict = PCA_decomposition(data_dict)

    return data_dict

def parse_dict(data_dict):

    key_list = ["time", "conc", "condition", "areas", "label", "perimeter", "day", "expr", "ending", "pixelSize", "pw_tot"]

    df = pd.DataFrame()
    for i in key_list:
        print(i)
        df[i] = np.array(data_dict[i])

    return df

def performPCA(df):
    pca = PCA(n_components=3,whiten = True)
    pca = pca.fit(df[df.keys()[7:]])
    components = pca.fit_transform(df[df.keys()[7:]])
    return pca, components
    
"""
def PCA_pipe(dict, hue_label):

    extra = np.array(['label', 'day', 'conc', 'time', 'runNum', 'expr', 'pixelSize'])

    headerR = np.array(list(map(lambda x: "r_" + x, dict["dfBins"][0].astype("str"))))
    headerPhi = np.array(list(map(lambda x: "p_" + x, dict["dfBins"][0].astype("str"))))
    
    tempDict = {}

    for i in range(51):
        tempDict[headerR[i]] = []
        tempDict[headerPhi[i]] = []
        
    for i in range(len(dict["dfBins"])):
        for j in range(51):
            tempDict[headerR[j]].append(dict["dfValueR"][i][j])
            tempDict[headerPhi[j]].append(dict["dfValuePhi"][i][j])

    df_pca = pd.DataFrame()

    for i in extra:
        df_pca[i] = dict[i]

    for i in tempDict.keys():
        df_pca[i] = np.array(tempDict[i])

    df_pca = df_pca.replace(np.nan, 0)

    fig, ax = plt.subplots(2, 3,figsize=(16, 10),facecolor='white')

    pca, components = performPCA(df_pca)
    
    comp = pd.DataFrame(components, columns = ['1','2', '3'])

    comp["label"] = df_pca["label"]
    comp["time"] = df_pca["time"]
    comp["pixelSize"] = df_pca["pixelSize"]

    total_var = pca.explained_variance_ratio_.sum() * 100

    ax[0,0].set_title("{} hours: explained {:.2f}%".format(i, total_var))
    ax[0,0].set_xlabel("1st component")
    ax[0,0].set_ylabel("2nd component")
    sns.scatterplot(x='1', y='2', data=comp, ax = ax[0,0], hue=hue_label)
    ax[0,0].legend(loc='upper right')

    ax[0,1].set_xlabel("2nd component")
    ax[0,1].set_ylabel("3rd component")
    sns.scatterplot(x='2', y='3', data=comp, ax = ax[0,1], hue=hue_label)
    ax[0,1].legend(loc='upper right')


    ax[0,2].set_xlabel("1st component")
    ax[0,2].set_ylabel("3rd component")
    sns.scatterplot(x='1', y='3', data=comp, ax = ax[0,2], hue=hue_label)
    ax[0,2].legend(loc='upper right')
    
    print('In {} hours, total Explained Variance: {}%'.format(i,total_var))

    weights = pca.components_
    
    ax[1,0].bar(np.arange(0,len(weights[0,:])),weights[0,:], label = "y-1", alpha = 0.5)
    ax[1,0].legend()

    ax[1,1].bar(np.arange(0,len(weights[0,:])),weights[1,:], label = "y-2", alpha = 0.5)
    ax[1,1].legend()

    ax[1,2].bar(np.arange(0,len(weights[0,:])),weights[2,:], label = "y-3", alpha = 0.5)
    ax[1,2].legend()
    ax[1,2].set_xlabel("Frequency [Hz]")

    return df_pca

def create_bins(lower_bound=0, width=1, quantity=50):

    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins

def find_bin(freq, amplitude, bins, values):
    for i in range(0, len(bins)):
        if bins[i][0] <= freq < bins[i][1]:
            values[i] += amplitude
            return values
    return values


def findFrequencies(data, j):
    
    xx = data[j][:,0,:]
    xx = (xx.astype(np.float32)-np.mean(xx,axis=0)[None,...])/np.std(xx,axis=0)[None,...]

    r = np.diff(scipy.ndimage.gaussian_filter(np.sqrt(xx[:,0]**2 + xx[:,1]**2),2))
    phi = np.diff(scipy.signal.detrend(scipy.ndimage.gaussian_filter(np.unwrap(np.arctan2(xx[:,0],xx[:,1])),2)))
    
    #r = np.diff(np.sqrt(xx[:,0]**2 + xx[:,1]**2))
    #phi = np.diff(np.unwrap(np.arctan2(xx[:,0],xx[:,1])))

    xxPolar = np.stack((r,phi), axis = 1)

    components = 50

    N = xxPolar.shape[0]
    fft_xy = np.fft.fft(xxPolar, axis = 0)
    idx = np.argsort(np.abs(fft_xy)**2,axis=0)[::-1]

    n = np.stack((idx[:components,0],idx[:components,1]),axis = 1)

    #i_sig = np.real(np.fft.ifft(sub_sig,axis=0))

    powerAll = np.abs(fft_xy)**2
    pw = np.zeros((components,2),dtype=np.complex128)
    pw[:,0] = powerAll[n[:,0],0]
    pw[:,1] = powerAll[n[:,1],1]

    bins = create_bins()

    valuesR = np.zeros(len(bins))
    valuesPhi = np.zeros(len(bins))

    for i in range(len(n)):    
        valuesR = find_bin(i,pw[i,0],bins,valuesR)
        valuesPhi = find_bin(i,pw[i,1],bins,valuesPhi)

    dfBins = np.arange(0,len(valuesR))
    
    return dfBins, valuesR, valuesPhi
    
"""