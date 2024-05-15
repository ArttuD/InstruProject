from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, QPoint

from PyQt6.QtGui import QImage

from PyQt6.QtWidgets import QApplication

from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap


import sys
import argparse
import sys
import numpy as np
import time
import os
import datetime
import ffmpeg
import cv2

import pandas as pd
import csv

from nd2reader import ND2Reader

from worker import Worker

class App(QWidget):

    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.path = args.path

        #UI geometry
        self.left = 0; self.top = 0
        self.width = 1000; self.height = 1000
        self.im_width = 720; self.im_height = 720


        self.ctr = {"track": False, "vector": False}
        self.worker = Worker(args.path, self.ctr)

        #cfg GUI
        self.initUI()

        self.clicked_clicks = []
        self.num_clicks = 0
        #self.cam.showProperties()

        self.c=0
        self.t=0
        self.z=0
        self.x=0
        self.y=0
        self.v=0
        self.cell_ID = 0

        self.video_path = "F:/instru_projects/TimeLapses/u-wells/collagen/240301_timelapses_collagen_3lines_48h_spheroidseeded.nd2"
        
        self.result_path = os.path.join(self.path, "results")

        self.createAndCheckFolder(self.result_path)

        self.dataDict = self.load_source()

        self.max_load = len(self.dataDict["file_name"])

        with ND2Reader(self.video_path) as images:
            self.meta = images.metadata

        self.z_stop = self.meta['z_levels'].stop
        self.t_stop = self.meta['num_frames']

        self.printLabel.setText("let's go!")

    def initUI(self):
        
        self.win = QWidget()
        self.styles = {"color": "#f00", "font-size": "20px"}
        self.win.resize(self.width,self.height)

        self.ctr = {"process": False, "track": False, "vector": False}

        #Main layout
        self.vlayout = QVBoxLayout()

        #1st row: Buttoms 
        self.hbutton = QHBoxLayout()

        #Text
        self.htext = QHBoxLayout()

        #navigation
        self.navbuttons = QHBoxLayout()

        self.hlabels = QHBoxLayout() 

        self.cfg_buttons() #cfg buttons

        #2nd row: field path 
        self.textLabel()
        self.cfg_image() #set label

        #Add final layout and display
        self.win.setLayout(self.vlayout)

        self.vlayout.addLayout(self.hbutton)
        self.vlayout.addLayout(self.htext)
        self.vlayout.addLayout(self.navbuttons)
        self.vlayout.addLayout(self.hlabels) 

        self.pen = QtGui.QPen()
        self.pen.setWidth(5)
        self.pen.setColor(QtGui.QColor("#EB5160")) 
        self.win.show()


    def track_painter(self, canvas, x, y):
    
        painter = QtGui.QPainter(canvas)
        painter.setPen(self.pen)

        painter.drawPoint(QPoint(x, y))
        painter.end()
        self.label.setPixmap(canvas)


    def load_source(self):

        data_dict = {"file_name": [], "channel":[], "ON": [], "location": [], "z-level": [],  "time_start": [] }

        with open(os.path.join(self.path, "Single_tracks.csv"), newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for count, row in enumerate(spamreader):
                if count == 0:
                    continue

                if len(row) < 6:
                    self.printLabel.setText("{} is incorrect".format(count))
                    continue
                
                for idx, i in enumerate(data_dict.keys()):
                    if i in ["location", "z-level", "time_start"]:
                        data_dict[i].append(int(row[idx])-1)
                    else:
                        data_dict[i].append(row[idx])

        return data_dict

    def cfg_image(self):
        """
        Create label, add accesories and label to layout
        """

        self.label = QLabel(self)
        self.set_black_screen()

        self.label.mousePressEvent = self.click_label
        self.hlabels.addWidget(self.label)
        self.hlabels.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.hlabels.setSpacing(0)



    def set_black_screen(self):

        background = np.zeros((self.im_width, self.im_height))
        h, w = background.shape
        bytesPerLine = 1 * w
        p = QImage(background,w, h, bytesPerLine, QImage.Format.Format_Grayscale8)
        self.label.setPixmap(QPixmap(QPixmap.fromImage(p)))  


    def textLabel(self):
        """
        File save path
        """

        self.file_field_Label = QLabel("Row File")
        self.file_field = QSpinBox()
        self.file_field.setFixedSize(int(self.width*1/7),50)
        self.htext.addWidget(self.file_field_Label)
        self.htext.addWidget(self.file_field)

        self.ID_field_Label = QLabel("Cell ID")
        self.ID_field = QSpinBox()
        self.ID_field.setFixedSize(int(self.width*1/7),50)
        self.htext.addWidget(self.ID_field_Label)
        self.htext.addWidget(self.ID_field)

        self.cell_ID = self.ID_field.value()

        self.textField = QTextEdit(self.args.path)
        self.textField.setFixedSize(int(self.width*3/7),50)

        self.printLabel = QLabel("Print Field")
   
        self.printLabel.setFixedSize(int(self.width*3/7),50)
        self.printLabel.setStyleSheet("background-color: white")


        self.htext.addWidget(self.textField)
        self.htext.addWidget(self.printLabel)

    def createAndCheckFolder(self,path):
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)


    def cfg_buttons(self):
        """
        Create and connect buttons, sliders, and check box
        """

        self.check_single = QCheckBox("single cell")
        self.check_single.stateChanged.connect(self.track_state)
        self.hbutton.addWidget(self.check_single)

        self.check_vector = QCheckBox("protrusion")
        self.check_vector.stateChanged.connect(self.vector_state)
        self.hbutton.addWidget(self.check_vector)

        #Start measurement
        self.btn_load = QPushButton("load")
        self.btn_load.pressed.connect(self.load)
        self.btn_load.setStyleSheet("background-color : green")
        self.hbutton.addWidget(self.btn_load)

        #Close Gui
        self.btn_save = QPushButton("save")
        self.btn_save.pressed.connect(self.save)
        self.btn_save.setStyleSheet("background-color : red")
        self.hbutton.addWidget(self.btn_save)
        
        #Close Gui
        self.btn_close = QPushButton("close")
        self.btn_close.pressed.connect(self.close)
        self.btn_close.setStyleSheet("background-color : red")
        self.hbutton.addWidget(self.btn_close)

        #Start measurement
        self.btn_z_down = QPushButton("<- z-level")
        self.btn_z_down.pressed.connect(self.z_down)
        self.btn_z_down.setStyleSheet("background-color : green")
        self.navbuttons.addWidget(self.btn_z_down)

        #Start measurement
        self.btn_z_up = QPushButton("z-level ->")
        self.btn_z_up.pressed.connect(self.z_up)
        self.btn_z_up.setStyleSheet("background-color : green")
        self.navbuttons.addWidget(self.btn_z_up)

        #Start measurement
        self.btn_t_down = QPushButton("<- t-level")
        self.btn_t_down.pressed.connect(self.t_down)
        self.btn_t_down.setStyleSheet("background-color : green")
        self.navbuttons.addWidget(self.btn_t_down)

        #Start measurement
        self.btn_t_up = QPushButton("t-level ->")
        self.btn_t_up.pressed.connect(self.t_up)
        self.btn_t_up.setStyleSheet("background-color : green")
        self.navbuttons.addWidget(self.btn_t_up)

    def load(self):

        self.clicked_clicks = []
        self.num_clicks = 0
        print(self.max_load, self.file_field.value())

        if self.file_field.value() >= (self.max_load):

            self.save()
            self.printLabel.setText("file Done!")
            self.file_field.setValue(int(0))

            self.c= int(self.dataDict["channel"][self.file_field.value()]) 
            self.t= int(self.dataDict["time_start"][self.file_field.value()])
            self.z= int(self.dataDict["z-level"][self.file_field.value()])
            self.x=0
            self.y=0
            self.v= int(self.dataDict["location"][self.file_field.value()]) 
            self.cell_ID = int(self.dataDict["ON"][self.file_field.value()])
        else: 
            #self.worker.path_frame =  self.dataDict["file_name"][self.file_field.value()]
            self.c= int(self.dataDict["channel"][self.file_field.value()]) 
            self.t= int(self.dataDict["time_start"][self.file_field.value()]) 
            self.z= int(self.dataDict["z-level"][self.file_field.value()]) 
            self.x=0
            self.y=0
            self.v= int(self.dataDict["location"][self.file_field.value()]) 
            self.cell_ID = int(self.dataDict["ON"][self.file_field.value()])

        self.ID_field.setValue(self.cell_ID)

        self.load_frame()
        

    def close(self):

        """
        Stop tuning, measurement or camera stream and reset Gui
        """

        self.btn_close.setStyleSheet("background-color : white")
        self.printLabel.setText("Closing, wait")

        self.btn_load.setStyleSheet("background-color : green")
        self.btn_close.setStyleSheet("background-color : red")

        self.set_black_screen()

        self.printLabel.setText("Ready for the new round!\nPlease remember change the path")
        exit(0)

    def track_state(self, state):

        if state == 2:
            self.ctr["track"] = True 
        else:
            self.ctr["track"] = False
    

    def vector_state(self, state):
        if state == 2:
            self.ctr["vector"] = True 
        else:
            self.ctr["vector"] = False
    
    def t_down(self):

        if self.t >0:
            self.t -= 1

        self.load_frame()

        return 0
    
    def t_up(self):

        if self.t < (self.t_stop):
            if ((self.ctr["track"]) |  (self.ctr["vector"])) & (self.t == (self.t_stop-1)):
    
                
                self.clicked_clicks = []
                self.num_clicks = 0

                self.file_field.setValue(self.file_field.value()+1)

                if self.file_field.value() >= (self.max_load):
                    self.save()
                    self.printLabel.setText("file Done!")
                    self.file_field.setValue(int(0))
                    self.c= int(self.dataDict["channel"][self.file_field.value()])
                    self.t= int(self.dataDict["time_start"][self.file_field.value()])
                    self.z= int(self.dataDict["z-level"][self.file_field.value()])
                    self.x=0
                    self.y=0
                    self.v= int(self.dataDict["location"][self.file_field.value()])
                    self.cell_ID = int(self.dataDict["ON"][self.file_field.value()])
                else:
                    #self.worker.path_frame =  self.dataDict["file_name"][self.file_field.value()]
                    self.c= int(self.dataDict["channel"][self.file_field.value()])
                    self.t= int(self.dataDict["time_start"][self.file_field.value()])
                    self.z= int(self.dataDict["z-level"][self.file_field.value()])
                    self.x=0
                    self.y=0
                    self.v= int(self.dataDict["location"][self.file_field.value()])
                    self.cell_ID = int(self.dataDict["ON"][self.file_field.value()])

                    self.ID_field.setValue(self.cell_ID)
            else:
                self.t += 1

        self.load_frame()
        return 0

    def z_down(self):

        if self.z > 0:
            self.z -= 1

        self.load_frame()

        return 0
    
    def z_up(self):
        if self.z < self.z_stop:
            self.z += 1

        self.load_frame()

        return 0
    
    def click_label(self, click):
        if click.button() == QtCore.Qt.MouseButton.LeftButton:

            self.track_painter(self.label.pixmap(), click.pos().x(),click.pos().y())
            self.num_clicks += 1
            self.clicked_clicks.append([click.pos().x(),click.pos().y()])

            if self.ctr["track"]:
                self.worker.update_points(click.pos().x(),click.pos().y(), self.ID_field.value(), self.file_field.value(), self.t, self.z, self.v)

            if (self.num_clicks == 2) :
                if (self.ctr["vector"]):

                    self.worker.update_vector(self.clicked_clicks, self.ID_field.value(), self.file_field.value(), self.t, self.z, self.v)
                    self.draw_vector(self.label.pixmap(), self.clicked_clicks)

                self.num_clicks = 0
                self.clicked_clicks = []

    def save(self):
        print("saving!")
        self.textField.setText("{}_{}_track.csv".format(self.path, id))
        self.worker.save_data(self.file_field.value(), self.ctr)

    def draw_vector(self, canvas, points):

        painter = QtGui.QPainter(canvas)
        painter.setPen(self.pen)
        painter.drawLine(points[0][0], points[0][1], points[1][0], points[1][1])
        painter.end()

        self.label.setPixmap(canvas)

    def load_frame(self):

        self.printLabel.setText("Loaded chan: {}, time {}, stack {}, loc: {}".format(self.c, self.t, self.z, self.v))

        with ND2Reader(self.video_path) as images:
            img = images.get_frame_2D(c=self.c, t=self.t, z=self.z, x=self.x, y=self.y, v=self.v)

        img = (img/256).astype("uint8")
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        h, w = img.shape
        bytesPerLine = 1 * w
        p = QImage(img,w, h, bytesPerLine, QImage.Format.Format_Grayscale8)
        p = p.scaled(self.im_width, self.im_height)

        self.label.setPixmap(QPixmap(QPixmap.fromImage(p)))  



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", default= "F:/instru_projects/TimeLapses/u-wells/collagen/results_240301_timelapses_collagen_3lines_48h_spheroidseeded", required= False ,help="path and name of output video")

    args = argParser.parse_args()

    app = QApplication(sys.argv)
    ex = App(args)
    sys.exit(app.exec())