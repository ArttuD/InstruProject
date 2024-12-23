
import glob
import os
import math as m
import json
from tools.func import *
from tools.saver import Cells
from nd2reader import ND2Reader
import argparse
import time


class Manager():

    def __init__(self, args):
        
        self.path = args.path
        
        self.video_name = os.path.split(self.path)[-1][:-4]
        self.root_path = os.path.split(self.path)[0]
        self.results = os.path.join(self.root_path, "results_{}".format(self.video_name))
        self.data_dict = {}
        day = self.video_name.split("_")[0]

        self.df = pd.DataFrame(self.create_dict())
        self.prev_dict = {}

        self.t_skip = 5
        
        with open('./dataStore/metalib.json', 'r') as f:
            self.own_meta = json.load(f)[day]

        self.focus_path = glob.glob(os.path.join(self.results, "corrected_*.pkl")) #""

        with open(self.focus_path[0], 'rb') as f:
            self.focus_dict = pickle.load(f)

        self.scaled_size = 1024
        self.converter = 2304/self.scaled_size
        self.v_init = 0
        self.t_init = 0

        self.matched_flag = True

        ret = self.check_logged()
        if ret == 1:
            print("Continue from the checkpoint")
        else:
            print("Starting a new set")

    def check_logged(self):

        log_files = glob.glob(os.path.join(self.results, "data_vector.csv"))

        if len(log_files) == 0:
            return -1
        else:
            self.df = pd.read_csv(log_files[0])

            v_max = np.max(self.df["location"].unique())
            df_sub = self.df[self.df["location"] == v_max]
            t_max = np.max(df_sub["time"].unique())

            df_sub = df_sub[df_sub["time"] == t_max]
            dics = df_sub.to_dict('records')

            for current in dics:

                current.pop('Unnamed: 0')
                label = int(current["cell_id"])
                self.prev_dict[label] = current
                
            self.object_num = np.max(df_sub["cell_id"].values)

            self.v_init = int(v_max)
            self.t_init = int(t_max + self.t_skip)

            return 1
        
    def vector_2d_length(self,v):
        return m.sqrt((v[0]-v[2]) ** 2 + (v[1]-v[3]) ** 2)

        
    def click_event(self, event, x, y, flags, params):
        
        if event == cv2.EVENT_LBUTTONDOWN:

            if (self.n_clicks == 0):
                ret = self.check_if_close(x*self.converter,y*self.converter)
                if ( ret != -1) & (self.matched_flag):
                    print("Found match: ", ret)
                    self.object_num = ret
                    self.matched_flag = False
                elif (ret == -1) & (self.matched_flag):
                    if len(self.data_dict.keys()) > 0:
                        for i in range(1000):
                            if (i not in self.data_dict.keys()) & (i not in self.df["cell_id"].values):
                                self.object_num = i
                                break
                    elif self.df.shape[0] > 1:
                        self.object_num = self.df["cell_id"].max() + 1
                    else:
                        self.object_num = 0

                    print("Did not find match: ", self.object_num, " with ", self.data_dict.keys())

                    self.data_dict[self.object_num] = self.create_dict()
                    self.data_dict[self.object_num]["cell_id"] = self.object_num
                    self.data_dict[self.object_num]["x"] = int(x*self.converter)
                    self.data_dict[self.object_num]["y"] = int(y*self.converter)
                    self.data_dict[self.object_num]["z"] = self.z_start
                    self.data_dict[self.object_num]["time"] = int(self.t)
                    self.data_dict[self.object_num]["location"] = int(self.v)

                    self.img_moc = cv2.circle(self.img_moc, (self.data_dict[self.object_num]["x"], self.data_dict[self.object_num]["y"]), radius=10, color=(0, 0, 255), thickness=4)
                    self.n_clicks += 1
                elif (self.matched_flag == False):
                        
                        print("Did not find match: ", self.object_num, " with ", self.data_dict.keys())
                        self.data_dict[self.object_num] = self.create_dict()
                        self.data_dict[self.object_num]["cell_id"] = self.object_num
                        self.data_dict[self.object_num]["x"] = int(x*self.converter)
                        self.data_dict[self.object_num]["y"] = int(y*self.converter)
                        self.data_dict[self.object_num]["z"] = self.z_start
                        self.data_dict[self.object_num]["time"] = int(self.t)
                        self.data_dict[self.object_num]["location"] = int(self.v)

                        self.img_moc = cv2.circle(self.img_moc, (self.data_dict[self.object_num]["x"], self.data_dict[self.object_num]["y"]), radius=10, color=(0, 0, 255), thickness=4)
                        self.n_clicks += 1
                        self.matched_flag = True

            elif (self.n_clicks == 1):

                self.data_dict[self.object_num]["x2"] = int(x*self.converter)
                self.data_dict[self.object_num]["y2"] = int(y*self.converter)
                self.data_dict[self.object_num]["z2"] = self.z_start

                self.data_dict[self.object_num]["length"] = self.vector_2d_length([self.data_dict[self.object_num]["x"],self.data_dict[self.object_num]["y"],int(x*self.converter),int(y*self.converter)])
                self.data_dict[self.object_num]["x_vec"] = int(x*self.converter) - self.data_dict[self.object_num]["x"]
                self.data_dict[self.object_num]["y_vec"] = int(y*self.converter) - self.data_dict[self.object_num]["y"]

                self.data_dict[self.object_num]["angle"] = np.arctan2(self.data_dict[self.object_num]["y_vec"], self.data_dict[self.object_num]["x_vec"])
                self.img_moc = cv2.arrowedLine(self.img_moc, (self.data_dict[self.object_num]["x"], self.data_dict[self.object_num]["y"]), (self.data_dict[self.object_num]["x2"], self.data_dict[self.object_num]["y2"]), (0, 0, 255)  , 5)
                self.n_clicks += 1

            elif (self.n_clicks == 2):
                self.wx = int(x*self.converter)
                self.wy = int(y*self.converter)
                self.img_moc = cv2.circle(self.img_moc, (int(x*self.converter),int(y*self.converter)), radius=5, color=(255, 0, 255), thickness=-1)
                self.n_clicks += 1

            elif (self.n_clicks == 3):
                self.data_dict[self.object_num]["width"] = self.vector_2d_length([int(x*self.converter),int(y*self.converter), self.wx, self.wy])
                self.img_moc = cv2.line(self.img_moc, (self.wx, self.wy), (int(x*self.converter), int(y*self.converter)), (255, 0, 255) , 5)
                self.n_clicks = 0

                print("Creted ", self.object_num, " moving on.")

            return 1

        elif event == cv2.EVENT_RBUTTONDOWN:

            if (len(self.data_dict.keys()) > 0) & (self.n_clicks==0):

                ret = self.find_closest(int(x*self.converter), int(y*self.converter))
                print(ret)
                last_key = list(self.data_dict)[ret]
                removed_tuple = self.data_dict.pop(last_key)
            elif (self.n_clicks>0) :
                last_key = list(self.data_dict)[-1]
                removed_tuple = self.data_dict.pop(last_key)
                self.object_num -= 1
                
            self.img_moc = self.img_bb.copy()
            self.re_draw()
            self.n_clicks = 0

        elif event == cv2.EVENT_MBUTTONUP:
            
            ret = self.check_if_close(int(x*self.converter), int(y*self.converter))
            if ret == -1:
                print("Nothing close")
            else:
                current = self.prev_dict[ret]
                self.data_dict[ret] = current
                self.data_dict[ret]["time"] = int(self.t)
                self.data_dict[ret]["location"] = int(self.v)
                self.img_moc = cv2.circle(self.img_moc, (int(current["x"]), int(current["y"])), radius=10, color=(0, 0, 255), thickness=4)
                self.img_moc = cv2.arrowedLine(self.img_moc, (int(current["x"]), int(current["y"])), (int(current["x2"]), int(current["y2"])), (0, 0, 255)  , 5)
        
        cv2.imshow("window", cv2.resize(self.img_moc, (self.scaled_size,self.scaled_size)) )

    def update_frame(self):

        for k_ in self.data_dict.keys():
            self.df = pd.concat((self.df, pd.DataFrame.from_dict([self.data_dict[k_]])))

        self.df.to_csv(os.path.join(self.results, "data_vector.csv"))

    def re_draw(self):

        if len(self.prev_dict.keys())>0: 

            for k_ in self.prev_dict.keys():

                current = self.prev_dict[k_]

                self.img_moc = cv2.circle(self.img_moc, (int(current["x"]), int(current["y"])), radius=10, color=(0, 0, 255), thickness=4)
                self.img_moc = cv2.arrowedLine(self.img_moc, (int(current["x"]), int(current["y"])), (int(current["x2"]), int(current["y2"])), (0, 255, 255)  , 5)
        
        if len(self.data_dict.keys())>0: 

            for k_ in self.data_dict.keys():

                current = self.data_dict[k_]
                self.img_moc = cv2.circle(self.img_moc, (int(current["x"]), int(current["y"])), radius=10, color=(0, 0, 255), thickness=4)
                self.img_moc = cv2.arrowedLine(self.img_moc, (int(current["x"]), int(current["y"])), (int(current["x2"]), int(current["y2"])), (0, 0, 255)  , 5)
        
        #df_sub = self.df[(self.df["location"] == self.v) & ( (self.df["time"] == (self.t - self.t_skip)) )].reset_index(drop = True)
        #print("Drawing")
        #print(self.df, "\n Sub",df_sub)
        #if df_sub.shape[0]>0:
        #    for i in range(df_sub.shape[0]):
        #        self.img_moc = cv2.circle(self.img_moc, (df_sub["x"].values[i], df_sub["y"].values[i]), radius=10, color=(0, 0, 255), thickness=4)
        #        self.img_moc = cv2.arrowedLine(self.img_moc, (df_sub["x"].values[i], df_sub["y"].values[i]), (df_sub["x2"].values[i], df_sub["y2"].values[i]), (0, 0, 255)  , 5)


    def check_if_close(self, x_clicked, y_clicked):

        min_distance = 1000
        idx = - 1

        df_sub = self.df[(self.df["location"].astype("int") == int(self.v)) & ( (self.df["time"].astype("int") == int(self.t - self.t_skip)))].reset_index(drop = True)

        #print("found: ", df_sub)
        if df_sub.shape[0]>0:

            for i in range(df_sub.shape[0]):
                distance = np.sqrt((df_sub["x"].values[i]-x_clicked)**2 + (df_sub["y"].values[i]-y_clicked)**2)
                if distance < min_distance:
                    min_distance = distance
                    idx = int(df_sub["cell_id"].values[i])

            if min_distance < 30:
                return idx
            else:
                return -1
        else:
            return -1

    def find_closest(self, x, y):

        dist = 1e6
        id = -1

        for count, k in enumerate(self.data_dict.keys()):

            x_n = self.data_dict[k]["x"]
            y_n = self.data_dict[k]["y"]
            eps = np.sqrt((x_n-x)**2 + (y_n-y)**2)

            if dist > eps:
                dist = eps
                id = count

        return id


    def create_dict(self):
        return {"cell_id": [], "time": [], "location": [], "x": [], "y": [], "z": [],"x2": [], "y2": [], "z2": [], "length":[], "x_vec":[], "y_vec":[], "angle":[], "width":[] }
        
        
    def pipe(self):

        # create a window
        cv2.namedWindow('window')
        # bind the callback function to window
        cv2.setMouseCallback('window', self.click_event)

        with ND2Reader(self.path) as images:

            self.metas = load_metadata(images)
            for d in range(len(self.metas["channels"])):
                if self.metas["channels"][d] == 'BF':
                    self.idx_bf = d
                elif self.metas["channels"][d] == 'Red':
                    self.idx_fl = d
            #self.idx_bf = 0
            #self.idx_fl = 0
            #self.metas = { "n_fields": 7, "n_frames": 25, "n_levels": 27}

            for v in range(self.v_init, self.metas["n_fields"]):

                #print("Processing location: ", v, "/", self.metas["n_fields"]-1)

                if self.v_init == v:
                    self.t = int(self.t_init)
                else: 
                    self.t = 0

                self.v = int(v)
                self.t = int(self.t)

                if (v in self.own_meta["ignore"]) | (v in self.own_meta["multi"]):
                    print("Location: ", v, " is on ignore or multi list")          
                    continue

                choosing_flag = True
                print("fetching: ", self.v, " / ", self.t)

                if self.v not in self.focus_dict.keys():
                    print("Location missing form the focus dict: v= ", self.v, ", file: ", os.path.split(self.path))
                    continue
                else:
                    if self.t < len(self.focus_dict[self.v]):
                        idx = self.focus_dict[self.v][self.t]
                    else:
                        print("Cannot find the timestamp from focus dictionary")
                        idx = 0
                        choosing_flag = False

                self.v_start = self.v            
                self.t_start = self.t
                self.z_start = idx
                self.n_clicks = 0

                if (idx == -1) | (idx == -2):
                    print("Labeled as overgrown or ended")
                    choosing_flag = False


                while (self.t<self.metas["n_frames"]) & (choosing_flag == True):
                                
                    self.img_moc = self.f_img(images)
                    self.img_bb = self.img_moc.copy()
                    self.re_draw()
                    cv2.imshow("window", cv2.resize(self.img_moc, (self.scaled_size,self.scaled_size)) )

                    # add wait key. window waits until user presses a key
                    kk = cv2.waitKey(0)

                    if (kk== 113):
                        if (self.n_clicks != 0):
                            windowText = "Finish clicking!".format()
                            cv2.putText(self.img_moc, windowText,(75, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow("window", cv2.resize(self.img_moc, (self.scaled_size,self.scaled_size)) )

                        else: #Exit q
                            self.t += self.t_skip
                            self.t_start = self.t
                            self.n_clicks = 0

                            self.update_frame()
                            self.prev_dict = self.data_dict
                            self.data_dict = {}
                    elif kk == 101: #clear e

                        self.t_start = self.t
                        self.n_clicks = 0
                        self.pts = []
                        self.object_num = 0

                        self.data_dict = {}
                        self.img_moc= self.f_img(images)
                        self.re_draw()

                    elif kk == 119: #Move z up w
                        if (self.n_clicks != 0):
                            print("finish clicking!")
                        else:
                            self.z_start += 1

                            if self.z_start == self.metas["n_levels"]:
                                self.z_start -=1

                            self.img_moc = self.f_img(images)
                            self.re_draw()

                    elif kk == 115: #Move z down s
                        if (self.n_clicks != 0):
                            print("finish clicking!")
                        else:
                            self.z_start -= 1
                            if self.z_start == -1:
                                self.z_start +=1

                            self.img_moc = self.f_img(images)
                            self.re_draw()
                    elif kk == 100: #Move up d
                        if (self.n_clicks != 0):
                            print("finish clicking!")
                        else:
                            self.t_start += 1
                            if self.t_start == self.metas["n_frames"]:
                                self.t_start -=1

                            self.img_moc = self.f_img(images)
                            self.re_draw()
                    elif kk == 97: #Move down a
                        if (self.n_clicks != 0):
                            print("finish clicking!")
                        else:
                            self.t_start -= 1
                            if self.t_start == -1:
                                self.t_start +=1

                        self.img_moc = self.f_img(images)
                        self.re_draw()
                    else:
                        print("incorrect key", kk)
                
                self.prev_dict = {}

    def f_img(self, img_env):
        print("fetching img: ", self.v_start, " / ", self.t_start)
        img_bf = img_env.get_frame_2D(c=int(self.idx_bf), t=int(self.t_start), z=int(self.z_start) , x=0, y=0, v=int(self.v_start))
        
        img_bf = (img_bf/(np.max(img_bf))*2**8).astype("uint8")
        img_bf = np.stack((img_bf, img_bf, img_bf), axis = -1)

        windowText = r"timestep: $\n$ t={}/{}, z={}/{}, v={}/{}".format( self.t_start, self.metas["n_frames"]-1, self.z_start, self.metas["n_levels"]-1, self.v_start, self.metas["n_fields"]-1)
        cv2.putText(img_bf, windowText, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        return img_bf
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Download results in the folder and ouputs results
                    """)
    parser.add_argument('--path','-p',required=False,default= None, help='Path to folder. eg. C:/data/imgs')

    #Save arguments
    args = parser.parse_known_args()[0]

    tracker = Manager(args)
    tracker.pipe()