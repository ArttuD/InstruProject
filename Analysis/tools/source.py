
import glob
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import skimage
from matplotlib.widgets import Slider

class opticalFlow():

    def __init__(self, args):

        #parse args
        self.path = args.path
        self.mode = self.mode

        self.saver = args.saver
        self.vis = args.vis

        self.savePath = args.save

        #defaults, read this form meta later
        self.pixelSize = 6.5/(20)
        self.width = 2304
        self.height = 2304

        self.out = None

        self.prvs = None
        self.ang_stack = None
        self.mag_stack = None

        self.saveDict = {"mag": [], "ang": []}
        self.saveDictVol = {"x": [], "y": [], "areas": [], "id": []}

        self.level = args.level
        self.channel = args.channel #TexasRed

        self.kernelOpen = np.ones((3,3),np.uint8)

        self.threshold = 99.75e-2
        self.time = 0

    def createrSaver(self):
        self.out = cv2.VideoWriter(os.path.join(self.savePath,'video_{}_{}_{}.avi'.format(self.mode,self.level,self.channel)), cv2.VideoWriter_fourcc('M','J','P','G'), 5, (750,750))

    def saveVideo(self, frame):
        self.out.write(frame)

    def pipe(self):

        root = glob.glob(os.path.join(self.path,"*.tif"))

        if self.saver:
            self.createrSaver()

        stackCounter = 0

        for j in range(len(root)): #len(root)

            parts = os.path.split(root[j])[1].split("_")
            time = parts[1]
            channel = parts[2]
            running = int(parts[3][:-4])
            frame = cv2.imread(root[j])#/256).astype('uint8')

            #print("Num ", int(running), "Channel ", channel, "number ", j )
            if (self.mode == "of") & (self.level == running) & (self.channel == channel):
                rep,img = self.opticalFlow(frame, stackCounter)
                stackCounter += 1
                if (self.vis) & (rep != -1) :
                    cv2.imshow("window",img)
                    k = cv2.waitKey(1) & 0xff
                    if k == ord("q"):
                        break
            elif (self.mode == "ve") & (self.level == running) & (self.channel == channel):
                if stackCounter==0:
                    self.tuneThreshold(frame)
                rep, img = self.volumeEstimate(frame, stackCounter)
                stackCounter += 1
                if (self.vis) & (rep != -1) :
                    cv2.imshow("window",img)
                    k = cv2.waitKey(1) & 0xff
                    if k == ord("q"):
                        break
            elif (self.mode == "vol") & (self.channel == channel):
                #if stackCounter==0:
                #    self.tuneThreshold(frame)
                
                rep, img = self.volumeFL(frame, stackCounter)
                print("Round", stackCounter)
                stackCounter += 1
                if (self.vis) & (rep != -1) :
                    cv2.imshow("window",img)
                    k = cv2.waitKey(1) & 0xff
                    if k == ord("q"):
                        break
            
            if stackCounter ==  20:
                stackCounter = 0
                self.time +=1

        print("exit loop")
        if self.saver:
            print("realease videoSaver")
            self.out.release()
            if self.mode == "of":
                #print("saving json")
                #self.saveJson(self.saveDict)
                self.saveNumpy(self.saveDict["ang"],  self.saveDict["mag"])
            else:
                self.saveJson(self.saveDictVol)
        
        if self.vis:
            cv2.destroyAllWindows()

        return self.saveDict, self.saveDictVol

    def opticalFlow(self, frame, stackNum):

        frameOrg = frame.copy()

        ang_stack = []
        mag_stack = []
    
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255

        if stackNum == 0:
            self.prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            return -1, None
        
        frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prvs,frameGray, None, 0.5, 10, 5, 3, 5, 1.5, 0)
        self.prvs = frameGray
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        if self.saver:
            if self.saveDict["mag"] == []:
                self.saveDict["mag"] = mag
                self.saveDict["ang"] = ang
            else: 
                self.saveDict["ang"] = np.concatenate([self.saveDict["ang"], mag], axis = 0)
                self.saveDict["mag"] = np.concatenate([self.saveDict["mag"], ang], axis = 0)
            

        if self.vis:
            readyFrame = self.visualize(ang, mag, rgb, frameOrg)
            return 1,readyFrame

        return -1, None

    def visualize(self,ang, mag, frameOpt, frameOrg):

        y = mag*np.sin(ang)
        x = mag*np.cos(ang) 

        fig, ax = plt.subplots(ncols=2,nrows = 2, figsize=(20, 15))

        _ = ax[0,0].imshow(cv2.cvtColor(frameOrg, cv2.COLOR_BGR2RGB))
        _ = ax[0,1].imshow(cv2.cvtColor(frameOpt, cv2.COLOR_BGR2RGB))
        _ = ax[1,0].imshow(mag, cmap='jet', interpolation='nearest')
        _ = ax[1,1].quiver(x[::10,::10], y[::10,::10], color='black', headwidth=1, scale = 50, headlength=4)
        _ = ax[0,0].title.set_text('Orginal Image')
        _ = ax[0,1].title.set_text('Masked Image')
        _ = ax[1,0].title.set_text('Optical flow map')
        _ = ax[1,1].title.set_text('Optical flow vector field')
        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        
        plt.close()
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (750,750))
        #print(img.shape)
        if self.saver:
            self.saveVideo(img)
        
        return img

    def volumeFL(self, frame, count):

        frame = cv2.GaussianBlur(frame,(3,3),0)
        frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(frameGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(image=th, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            for counter,k in enumerate(contours):
                currentArea = cv2.contourArea(k) 
                #print(currentArea)
                if currentArea > 11000:
                    M = cv2.moments(k)
                    cv2.drawContours(image=frame, contours=contours, contourIdx=counter, color=(0, 255, 0), thickness=3)
                    if self.saveDictVol["x"] == []:
                        self.saveDictVol["x"] = np.array([int(M['m10']/M['m00'])])
                        self.saveDictVol["y"] = np.array([int(M['m01']/M['m00'])])
                        self.saveDictVol["areas"] = np.array([cv2.contourArea(k)])
                        self.saveDictVol["id"] = np.array(["time_{}_stamp_{}_running_{}".format(self.time,count, counter)])
                    else: 
                        #print("len ", len(contours), " x ", self.saveDictVol["x"], " center ", int(M['m10']/M['m00']))
                        self.saveDictVol["x"] = np.concatenate([self.saveDictVol["x"], np.array([int(M['m10']/M['m00'])])], axis = 0)
                        self.saveDictVol["y"] = np.concatenate([self.saveDictVol["y"],  np.array([int(M['m01']/M['m00'])])], axis = 0)
                        self.saveDictVol["areas"] = np.concatenate([self.saveDictVol["areas"],  np.array([cv2.contourArea(k)])], axis = 0)
                        self.saveDictVol["id"] = np.concatenate([self.saveDictVol["id"],  np.array(["time_{}_stamp_{}_running_{}".format(self.time,count, counter)])], axis = 0)
        
        if self.vis:
            img = self.visualizeVolume(count, frame, th, np.zeros_like(frame), self.saveDictVol["areas"])

        return 1, img

    def volumeEstimate(self, frame, count):

        self.kernelOpen = np.ones((3,3),np.uint8)
        frameOrg = frame.copy()

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgMask = skimage.filters.gaussian(frameGray,5)
        
        fgMask = skimage.filters.farid(fgMask)
        fgMask = 1-fgMask<self.threshold #99.925e-2
        opening = fgMask.astype("uint8")

        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, self.kernelOpen)
        opening = skimage.morphology.closing(opening, skimage.morphology.disk(10))
        e2 = skimage.morphology.white_tophat(opening,skimage.morphology.disk(5))
        e2 = skimage.morphology.dilation(e2,skimage.morphology.disk(5))
        
        tmp = opening.astype("int")-e2.astype("int")
        tmp[tmp<0] = 0
        tmp = tmp.astype("uint8")
        
        contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            maxCnt = sorted(contours, key=cv2.contourArea, reverse= True)
            for counter,k in enumerate(maxCnt):
                currentArea = cv2.contourArea(k) 
                #print(currentArea)
                if currentArea > 11000:
                    M = cv2.moments(k)
                    if self.saveDictVol["x"] == []:
                        self.saveDictVol["x"] = np.array([int(M['m10']/M['m00'])])
                        self.saveDictVol["y"] = np.array([int(M['m01']/M['m00'])])
                        self.saveDictVol["areas"] = np.array([cv2.contourArea(k)])
                        self.saveDictVol["id"] = np.array(["stamp_{}_running_{}".format(count, counter)])
                    else: 
                        #print("len ", len(contours), " x ", self.saveDictVol["x"], " center ", int(M['m10']/M['m00']))
                        self.saveDictVol["x"] = np.concatenate([self.saveDictVol["x"], np.array([int(M['m10']/M['m00'])])], axis = 0)
                        self.saveDictVol["y"] = np.concatenate([self.saveDictVol["y"],  np.array([int(M['m01']/M['m00'])])], axis = 0)
                        self.saveDictVol["areas"] = np.concatenate([self.saveDictVol["areas"],  np.array([cv2.contourArea(k)])], axis = 0)
                        self.saveDictVol["id"] = np.concatenate([self.saveDictVol["id"],  np.array(["stamp_{}_running_{}".format(count, counter)])], axis = 0)
                    

                    cv2.drawContours(image=frameOrg, contours=maxCnt, contourIdx=counter, color=(0, 255, 0), thickness=3)
        
        if self.vis:
            img = self.visualizeVolume(count, frameOrg, fgMask, tmp, self.saveDictVol["areas"])

        return 1, img
            

    def visualizeVolume(self,count, contoured, fgMask, tmp, area):

        fig, ax = plt.subplots(ncols=2,nrows = 2, figsize=(20, 15))

        _ = ax[0,0].imshow(cv2.cvtColor(contoured, cv2.COLOR_BGR2RGB))
        _ = ax[0,1].imshow(tmp)
        _ = ax[1,0].imshow(fgMask)
        #_ = ax[1,1].plot(np.arange(0,count+1), area[-1])
        _ = ax[0,0].title.set_text('Orginal Image')
        _ = ax[0,1].title.set_text('fardi filter')
        _ = ax[1,0].title.set_text('after morphology')
        _ = ax[1,0].title.set_text('Area')
        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        
        plt.close()
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (750,750))
        #print(img.shape)
        if self.saver:
            self.saveVideo(img)
        
        return img
    
    def tuneThreshold(self, frame):

        def processImg(frame, thVal):
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgMask = skimage.filters.gaussian(frameGray,5)
            
            fgMask = skimage.filters.farid(fgMask)
            fgMask = 1-fgMask<thVal #99.925e-2
            opening = fgMask.astype("uint8")

            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, self.kernelOpen)
            opening = skimage.morphology.closing(opening, skimage.morphology.disk(10))
            e2 = skimage.morphology.white_tophat(opening,skimage.morphology.disk(5))
            e2 = skimage.morphology.dilation(e2,skimage.morphology.disk(5))
            
            tmp = opening.astype("int")-e2.astype("int")
            tmp[tmp<0] = 0
            tmp = tmp.astype("uint8")

            return tmp

        tmp = processImg(frame, self.threshold)

        fig, ax = plt.subplots(ncols=2,nrows = 1, figsize=(20, 15))
        ax[0].imshow(frame)
        HANDLE = ax[1].imshow(tmp)

        # Make a horizontal slider to control the frequency.
        pos = fig.add_axes([0.25, 0.1, 0.65, 0.03])

        Th_slider = Slider(
            ax=pos,
            label='Farag Filter',
            valmin=98.00e-2,
            valmax=1,
            valinit=self.threshold,
        )

        def update(val):
            tmp =  processImg(frame, Th_slider.val)
            HANDLE.set_data(tmp)
            fig.canvas.draw_idle()

        Th_slider.on_changed(update)
        
        
        plt.show(block=True)
        

        self.threshold = Th_slider.val

    def saveJson(self,saveFile):
        print("In saver")

        for i in saveFile.keys():
            saveFile[i] = saveFile[i].tolist()

        print("dumbing")
        # Serializing json
        json_object = json.dumps(saveFile, indent=4)
        
        print("saving Json")
        # Writing to sample.json
        
        with open(os.path.join(self.savePath,'data_{}_{}_{}.json'.format(self.mode,self.level,self.channel)), "w") as outfile:
            outfile.write(json_object)

    def saveNumpy(self, angles, magnitudes):
        np.save(os.path.join(self.savePath,'angles_{}_{}_{}.npy'.format(self.mode,self.level,self.channel)), angles)
        np.save(os.path.join(self.savePath,'magnitude_{}_{}_{}.npy'.format(self.mode,self.level,self.channel)), magnitudes)
        
