import cv2
import numpy as np
from tools.unet import Unet,Unet_ResNet,Unet_plusplus
from sklearn.cluster import DBSCAN
from tools.losses import focal_loss
import tensorflow as tf

class Detector:
    def __init__(self,path,threshold):
        self.frame = None
        self.imgDraw = None
        self.threshold = float(threshold)
        self.init_model()
        self.path = path

    def init_model(self):
        custom_objects = {"loss": focal_loss}
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.model = tf.keras.models.load_model('./tools/arttu_model5')

    def tracker(self,imgFrame):
        #Receive images: 1) track (fluorecent) and 2) draw (overlap)
        self.frame = imgFrame

        #Preprocess the frame
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)[...,np.newaxis]
        gray = np.tile(gray,(1,1,3)).astype(np.float32)
        gray = 1.0-gray/np.max(gray)
        gray = cv2.resize(gray,(512,512))

        #detect
        res = self.model.predict(gray[np.newaxis,...])[0]

        #Save corodinates
        out = np.zeros((*res.shape[:2],3)).astype(np.uint8)
        out[...,0] = ((res>self.threshold)*255).astype(np.uint8)[...,0]
        x,y,z = np.where(res>self.threshold)
        
        z = (z+1)

        return x,y,z