
import glob
import os
import json
import cv2
import numpy as np
import pickle
import skimage

from sklearn.cluster import DBSCAN

from tools.detector import Detector
from tools.MTT.track_manager import TrackManager


class Process:

    def __init__(self, args):
        
        self.savePath = args.savePath
        self.path = args.path
        self.saver = args.save
        self.metaData = {}

        self.detections = []
        self.dets = []
        self.boxLabels = []

        self.width = None
        self.height = None

        self.vizLevel = 8

        self.vis = args.vis
        self.timer = 0

        self.out = None
        self.frame = np.zeros((2305,2305))

        try: 
            self.readMeta()
        except:
            print("No metadata available")
            self.metaData["pixelSize"] = 6.5/(20*0.63)
            self.metaData["Prefix"] = 0
            self.metaData["z-step_um"] = 10
            self.metaData["Slices"] = 21
            self.width = 2304
            self.height = 2304

        self.detector = Detector(args.path,args.threshold)
        self.tracker = TrackManager(min_count=6,max_count = 2, gating = 75)

    def displayFrame(self):
            # Displaying the image
            cv2.imshow("main", cv2.resize(self.frame,(512,512)))
            k = cv2.waitKey(1); 
            if k == 'q':
                cv2.destroyAllWindows()

    def writeFrame(self):
        self.out.write(self.frame)

    def preProcess(self, frame):
        try:
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_adapteq = skimage.exposure.equalize_adapthist(imgGray, clip_limit=0.005)
            pImg = np.stack([img_adapteq,img_adapteq,img_adapteq], axis = -1)

            return pImg.astype("uint8")
        except:
            return frame

    def readVideo(self):
        
        root = glob.glob(os.path.join(self.path,"*.tif"))

        if self.saver:
            self.out = cv2.VideoWriter("videoTracked_{}.avi".format(self.metaData["Prefix"]), cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.width,self.height))

        stackCounter = 0
        for j in range(len(root)): #len(root)
            
            parts = os.path.split(root[j])[1].split("_")
            time = parts[1]
            channel = parts[2]
            running = int(parts[3][:-4])
            frame = cv2.imread(root[j])#/256).astype('uint8')


            if ("Refl-Dic" == channel):
                #if stackCounter == self.vizLevel:
                self.frame = frame
                
            elif ("TexasRed" == channel):
                frame = self.preProcess(frame)
                stackCounter = self.processTR(frame, stackCounter)

        if self.saver:
            self.out.release()
            self.savePickle()

        if self.vis:
            cv2.destroyAllWindows()

        return self.metaData, self.tracker

    def savePickle(self):
        with open(f"{self.savePath}/trackerOutput.pickle", 'wb') as outp:
            pickle.dump(self.tracker, outp, pickle.HIGHEST_PROTOCOL)
            print("saved output pickle")

    def processTR(self, frame, stackCounter):
        #frame = self.preProcessFl(frame)

        print("Process stack ", stackCounter)

        x,y,z = self.detector.tracker(frame)
        x = x*self.metaData["pixelSize"]; y = y*self.metaData["pixelSize"]
        z = z*self.metaData["z-step_um"]

        if self.detections == []:
            self.detections = np.stack((np.asarray(x),np.asarray(y),np.asarray(z)),axis = -1)
            
        else:
            self.detections = np.concatenate([self.detections,np.stack((np.asarray(x),np.asarray(y),np.asarray(z)),axis = -1)], axis = 0)
            print(self.detections.shape)

        if stackCounter == self.metaData["Slices"]-1:
            #Combine to one array
            #self.detections = np.concatenate(self.detections, axis = 0)
            print("stack counter ", self.detections.shape)            

            #Cluster based on the location
            self.cluster()

            print("processed one stack, computing clusters from array")
            self.trackerCommunicator()
            if self.vis:
                self.displayFrame()
            if self.saver:
                self.writeFrame()
                
            stackCounter = self.clean()
            print("stamp done:\n", self.timer) 
            self.timer += 1
        else:
            stackCounter +=1
        return stackCounter
        
    def clean(self):
        self.dets = []
        self.boxLabels = []
        self.detections = []

        return 0

    
    def cluster(self):

        if len(self.detections) == 0:
            return
        
        clustering = DBSCAN(eps=30, min_samples=3).fit(self.detections)
        n_unique = np.unique(clustering.labels_)
        
        for i in n_unique:
            if i==-1:
                continue

            cond = clustering.labels_==i
            coords = self.detections[cond].mean(axis=0)
            X_ = self.detections[cond]

            if X_.shape[0]>2:
                self.dets.append([coords[0],coords[1],coords[2],1,1])
                self.boxLabels.append(0) #Detect only cells

    def trackerCommunicator(self):

        self.tracker.update(np.stack(self.dets), self.timer,self.boxLabels)

        all_tracks = self.tracker.trackers
        for track in all_tracks:
            if (not track.killed):
                t_col = (255,0,0)
                m_col = (255,64,64)
                if track.tentative:
                    t_col = (100,100,100)
                    m_col = (39,64,139)
                track_info = (np.array(track.history[-5:])[:,:2]).reshape((-1, 1, 2))
                print(track_info)
                #self.image = cv2.polylines(self.image, [track_info], True, t_col, 2)
                if (self.vis) & (track_info[-1,0,0] != track_info[-1,0,1]):
                    self.frame = cv2.drawMarker(self.frame, (int(track_info[-1,0,1]/(self.metaData["pixelSize"]*512/self.frame.shape[0])),int(track_info[-1,0,0]/(self.metaData["pixelSize"]*512/self.frame.shape[1]))), m_col, 0, 50, 5)
       
    def readMeta(self):
        root = glob.glob(os.path.join(self.path,"*.txt"))
        # Opening JSON file
        f = open(root[0])
        # returns JSON object as 
        # a dictionary
        data = json.load(f)    
        data = data
        self.metaData = data["Summary"]

        self.metaData["pixelSize"] = 6.5/(20*0.63)

        self.width = self.metaData["Width"] 
        self.height = self.metaData["Height"] 

        print("Meta found and downloaded successfully!")




