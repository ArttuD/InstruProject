from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal

from PyQt6.QtCore import QPoint, pyqtSignal, QThread, pyqtSlot as Slot
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

from worker import Worker

class App(QWidget):

    def __init__(self, args):
        super().__init__()

        #Control threads
        self.ctrl = {}
        
        self.args = args
        self.path = args.path

        #UI geometry
        self.left = 0; self.top = 0
        self.width = 1000; self.height = 1000
        self.im_width = 720; self.im_height = 720

        self.process_flag = False

        #cfg GUI
        self.initUI()


        self.clicked_clicks = []
        self.num_clicks = 0
        #self.cam.showProperties()

        self.worker = Worker(self.ctr, args.path)


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

        self.ID_field = QSpinBox()
        self.ID_field.setFixedSize(int(self.width*1/7),50)

        self.textField = QTextEdit(self.args.path)
        self.textField.setFixedSize(int(self.width*3/7),50)

        self.printLabel = QLabel("Print Field")
   
        self.printLabel.setFixedSize(int(self.width*3/7),50)
        self.printLabel.setStyleSheet("background-color: white")

        self.htext.addWidget(self.ID_field)
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
        self.check_single.stateChanged.connect(self.check_single_func)
        self.hbutton.addWidget(self.check_single)

        self.check_vector = QCheckBox("protrusion")
        self.check_vector.stateChanged.connect(self.check_vector_func)
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

        #track
        self.track = QCheckBox("Track")
        self.track.stateChanged.connect(self.track_state)
        
        #track
        self.vector = QCheckBox("vector")
        self.vector.stateChanged.connect(self.vector_state)

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

        """
        Start measurement
            -Fetch path
            -start current driver and camera
        """

        self.btnStart.setStyleSheet("background-color : white")

        self.process_flag = True

        self.createAndCheckFolder(self.textField.toPlainText())

        self.printLabel.setText("Measurement started")
        self.process.start()
        

    def close(self):

        """
        Stop tuning, measurement or camera stream and reset Gui
        """

        self.btn_close.setStyleSheet("background-color : white")
        self.printLabel.setText("Closing, wait")

        if self.process_flag:
            print("test")

        self.btn_load.setStyleSheet("background-color : green")
        self.btn_close.setStyleSheet("background-color : red")

        self.process_flag = False
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
        self.read()
        return 0
    
    def t_up(self):
        return 0

    def z_down(self):
        return 0
    
    def z_up(self):
        return 0
    
    def click_label(self, click):
        if click.button() == QtCore.Qt.MouseButton.LeftButton:
            print(click.pos().x(),click.pos().y())
            self.track_painter(self.label.pixmap(), click.pos().x(),click.pos().y())

            self.num_clicks += 1
            self.clicked_clicks.append([click.pos().x(),click.pos().y()])

            if self.num_clicks == 2:
                self.draw_vector(self.label.pixmap(), self.clicked_clicks)
                self.num_clicks = 0
                self.clicked_clicks = []

    def save(self):
        self.worker.save_data()

    def draw_vector(self, canvas, points):

        painter = QtGui.QPainter(canvas)
        painter.setPen(self.pen)
        painter.drawLine(points[0][0], points[0][1], points[1][0], points[1][1])
        painter.end()

        self.label.setPixmap(canvas)


    def check_single_func(self, state):
        if state == 0:
            self.ctr["single"] = False
        else:
            self.ctr["single"] = True


    def check_vector_func(self, state):
        if state == 0:
            self.ctr["vector"] = False
        else:
            self.ctr["vector"] = True


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", default= "./", required= False ,help="path and name of output video")

    args = argParser.parse_args()

    app = QApplication(sys.argv)
    ex = App(args)
    sys.exit(app.exec())