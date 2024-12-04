

import numpy as np
from cellpose import core, io, models, denoise
from glob import glob
import cv2
from tools.func import *
from sklearn.cluster import DBSCAN
from nd2reader import ND2Reader
from tools.MTT.track_manager import TrackManager


import argparse

class manager():

    def __init__(self, args):

        self.path = args.path
        self.track_flag = args.track

        self.model_path = "C:/Users/lehto/git/InstruProject/NikonPipes/dataStore/img_store/models/model_final_arttu"
        self.meta_path = "./dataStore/metalib.json"

        vid_parts = os.path.split(self.path)
        self.results = os.path.split(self.path)[0] + "/results_{}".format(vid_parts[1][:-4])
        self.segments_path =  vid_parts[0] + "/results_{}/{}_corrected_detections.pkl".format(vid_parts[1][:-4], vid_parts[1][:-4])
        self.focus_path =  vid_parts[0] + "/results_{}/corrected_focus_indixes.pkl".format(vid_parts[1][:-4])
        self.day = os.path.split(self.path)[1].split("_")[0]

        self.chan = "Grayscale"
        self.chan2 = "None"

        self.detect = detector(self.model_path, [self.chan, self.chan2])
        try:
            with open(self.segments_path, 'rb') as f:
                self.segment_dict = pickle.load(f)
        except:
            print("No detections!")
            exit()
        try:
            with open(self.focus_path, 'rb') as f:
                self.focus_dict = pickle.load(f)
        except:
            print("No focus!")
            exit()

        try:
            with open('./dataStore/metalib.json', 'r') as f:
                self.own_meta = json.load(f)
                self.own_meta = self.own_meta[self.day]
        except:
            print("No metalib!")
            exit()   



    
    def create_video(self, v):
        self.out_name = self.results + '/detector_{}_{}.mp4'.format( self.day, v )
        self.out_process = cv2.VideoWriter(self.out_name , cv2.VideoWriter_fourcc(*"mp4v"), 5, (2304,2304))   
    
    def pipe(self):


        with ND2Reader(self.path) as images:
            #c -channel, v - locations, t - time
            
            self.metas = load_metadata(images)

            for d in range(len(self.metas["channels"])):
                if self.metas["channels"][d] == 'BF':
                    idx_bf = d
                elif self.metas["channels"][d] == 'Red':
                    idx_fl = d

            #images.iter_axes = "vtc"
            #images.bundle_axes = "zyx"


            total_dets = []
            v_id = 0


            for v_id in range(self.metas["n_fields"]):
                t_id = 0
                img_prev = np.ones((2304,2304), dtype=np.uint16)
                self.create_video(v_id)
                if (v_id in self.own_meta["ignore"]) | (v_id in self.own_meta["multi"]):
                    self.end_location(t_id, v_id,total_dets)
                    img_prev = np.ones((2304,2304), dtype=np.uint16)          
                    continue
                elif v_id == self.metas["n_fields"]:
                    print("processing done")
                    break
                
                for t_id in range(self.metas["n_frames"]):

                    focus_idx = self.focus_dict[v_id][t_id]

                    if (focus_idx == -1) | (focus_idx == -2):
                        self.end_location(t_id, v_id,total_dets)
                        img_prev = np.ones((2304,2304), dtype=np.uint16)
                        break

                    big_idx = self.segment_dict["loc_{}_ch_1".format(v_id)]['big_idx'][t_id]
                    spheroid = self.segment_dict["loc_{}_ch_1".format(v_id)]['mask'][t_id][big_idx]

                    img_stack=[]

                    for z_id in range(self.metas['n_levels']):
                        img_stack.append(images.get_frame_2D(c=idx_fl, t=int(np.abs(t_id)), z=int(np.abs(z_id)), x=0, y=0, v=int(np.abs(v_id))))

                    masks = self.detect.detect(img_stack)
                    detections, labels = self.detect.process_masks(masks, v_id, t_id, spheroid)

                    if len(detections) > 0:
                        total_dets.append(pd.DataFrame(np.array(detections), columns=["x","y","z","location","time"]))
                    else:
                        print("No detections from {} {}".format(v_id, t_id))

                    try:
                        img_bf = images.get_frame_2D(c=int(idx_bf), t=int(np.abs(t_id)), z=int(np.abs(focus_idx)), x=0, y=0, v=int(np.abs(v_id)))
                        img_prev = img_bf.copy()
                    except:
                        print("Cannot fetch brightfield frame. Ending location!")
                        img_bf = img_prev.copy()

                    img_bf = self.finish_stack(img_bf, detections, spheroid)

                    ##plt.imshow(img_bf)
                    ##plt.show(block=False)
                    ##plt.pause(10)
                    ##plt.close()

                    self.out_process.write(img_bf)
                    print("Process fields: {}/{}, time {}/{}".format(v_id, self.metas["n_fields"]-1, t_id, self.metas["n_frames"]-1))

                self.end_location(t_id, v_id, total_dets)
                img_prev = np.ones((2304,2304), dtype=np.uint16)
                total_dets = []

            print("processing done")
        
        self.finish_detections()

        if self.track_flag:
            self.init_tracker()

    def init_tracker(self):
        tracker = Tracker(self.results)
        tracker.pipe()

    
    def end_location(self, t, v, datas):
        print("Done with location", v, "moving on")
        v += 1
        self.out_process.release()

        if len(datas)>0:
            datas = pd.concat(datas)
            datas["cell_id"] = np.arange(datas.shape[0])
            datas.to_csv(self.results + '/detector_{}_{}.csv'.format( self.day, v-1 ))
        else:
            os.remove(self.out_name)

        if v != self.metas["n_fields"]:
            self.create_video(v)

    def finish_stack(self, img, dets, segment):

        img_n = np.zeros_like(img)
        cv2.normalize(img,  img_n, 0, 255, cv2.NORM_MINMAX)
        img_n = img_n.astype("uint8")
        img_n = np.stack((img_n,img_n,img_n),axis = -1)

        cv2.drawContours(img_n, [segment], -1, (0,255,0), 3)

        for i in dets:
            cv2.drawMarker(img_n, (int(i[0]), int(i[1])),(255,0,0), markerType=cv2.MARKER_STAR, 
                    markerSize=30, thickness=4, line_type=cv2.LINE_AA)
            
        return img_n
    
    def finish_detections(self):

        paths_ = glob(os.path.join(self.results,"detector_*.csv"))
        df_array = []

        i = 0
        for i in paths_:
            cPath = pd.read_csv(i, encoding='unicode_escape')
            df_array.append(cPath[cPath.columns[1:]])


        df_array = pd.concat(df_array)
        df_array.to_csv(self.results + '/data_track.csv')
        print("Saved in location", self.results + '/data_track.csv')

        for i in paths_:
            os.remove(i)


class detector():

    def __init__(self, model_path, channels):

        self.channels = channels
        self.model = models.CellposeModel(gpu=True, pretrained_model=model_path)

        self.diameter =  50#self.model.diam_labels
        self.dn = denoise.DenoiseModel(gpu=True, model_type="deblur_cyto3",diam_mean=self.diameter)

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def detect(self, img_stack):

        imgs_ = np.array(self.dn.eval(list(img_stack), channels=self.channels, diameter=self.diameter ))
        imgs_dn = []

        for num, c_img in enumerate(imgs_):
            imgs_dn.append((c_img).astype("float"))

        # run model on test images
        masks, flows, styles = self.model.eval(imgs_dn, 
                                        channels=self.channels,
                                        diameter=self.diameter,
                                        normalize=True
                                        )
        
        return masks
    
    def process_masks(self, masks, v, t, cnt_probe):
        coords = []
        dets = []
        boxLabels = []

        for num,mask in enumerate(masks):

            contours, hierarchy = cv2.findContours(image=mask.astype("uint8"), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

            if len(contours) == 0:
                #print("No masks detected: {} {} {}".format( v, t, num))
                continue
            else:
                for nmr, cnt in enumerate(contours):

                    area = cv2.contourArea(cnt)
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    loc_flag  = cv2.pointPolygonTest(cnt_probe, (x, y), False)

                    if (loc_flag == -1):
                        #print("Accepting: Area {}, detected: {} {} {}".format(area, v, t, num))
                        coords.append([ v, t, x, y, num, area])
                    #else:
                        #print("Ignoring: Area {}, detected: {} {} {}".format(area, v, t, num))

        if len(coords)>0:
            coords = np.stack(coords, axis = 0)
            scan = np.stack((coords[:,2],coords[:,3],coords[:,4]), axis = -1)
            clustering = DBSCAN(eps=50, min_samples=3).fit(scan)
            n_unique = np.unique(clustering.labels_)

            for i in n_unique:
                if i == -1:
                    continue

                cond = clustering.labels_== i
                coords_ = scan[cond].mean(axis=0)
                X_ = scan[cond]

                if X_.shape[0]>0:

                    dets.append([coords_[0],coords_[1],coords_[2],v, t])
                    boxLabels.append(0)
            

        return dets, boxLabels          

class Tracker():

    def __init__(self, result_path):

        self.result_path = result_path
        self.day = os.path.split(result_path)[1].split("_")[1]
        self.df = pd.read_csv(self.result_path + '/data_track.csv')
        #self.df = self.df.drop(["Unnamed: 0"]).reset_index(drop=True)

    def pipe(self):

        for tags, data_location in self.df.groupby(["location"]):
            self.tracker_obj = TrackManager(min_count=5,max_count=6, gating = 150, gating_= 200)

            data_location = data_location.reset_index(drop = True)
            tags = tags[0]

            v = int(tags)
            path_in_ = os.path.join(self.result_path, os.path.split(self.result_path)[1][8:])
            path_in_ = path_in_ + '_{}_*.mp4'.format(int(v))
            paths_ = glob(path_in_, recursive=True)

            out_name = paths_[0] #self.result_path + '/{}_timelapse_IPN3mM_3lines_48h_comments_{}_*.mp4'.format( self.day, v ) 

            rec_count = 0
            cap = cv2.VideoCapture(out_name)

            if not cap.isOpened():
                print("Cannot open camera")
                exit()

            frames = []
            for tags_, data_stamp in data_location.groupby("time"):

                time = int(tags_)
                while rec_count<=time:
                    ret, frame = cap.read()
                    frames.append(frame)
                    rec_count += 1
                

                data_stamp = data_stamp.reset_index(drop = True)
                data_stamp["labels"] = 1
                data_stamp["dummy"] = 1
                dets = data_stamp[["x", "y", "z", "labels", "dummy"]].values
                self.tracker_obj.update(dets, tags_, np.zeros(data_stamp.shape[0])) 
                img = self.draw_tracks(self.tracker_obj.trackers, frames[-1])
                frames[-1] = img 


                #plt.imshow( frames[-1])
                #plt.show()

            print("Processed location {}".format(v))

            cap.release()

            self.out_name = self.result_path + '/detector_{}_{}.mp4'.format( self.day, v )
            self.out_process = cv2.VideoWriter(self.out_name , cv2.VideoWriter_fourcc(*"mp4v"), 5, (2304,2304)) 

            for nrm, frame in enumerate(frames):
                #frame = np.stack((frame, frame, frame), axis = -1)
                self.out_process.write(frame)

            self.out_process.release()
        
        print("All processed")

    def draw_tracks(self, all_tracks, img):

        for track in all_tracks:

            if (not track.killed):

                t_col = (255,0,0)
                m_col = (255,0,255)

                if track.tentative:
                    t_col = (100,100,100)
                    m_col = (100,0,100)

                track_info = (np.array(track.history[-5:])[:,:2]).reshape((-1, 1, 2))
                
                #if track_info.shape[0] > 1:


                if ((track_info[-1,0,0] != track_info[-1,0,1]) ):

                    if track.tentative:
                        continue

                    img = cv2.polylines(img, np.int32([track_info]), True, t_col, 2)
                    img = cv2.drawMarker(img, (int(track_info[-1,0,0]),int(track_info[-1,0,1])), m_col, 0, 50, 5)

        return img


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
        description="""Download results in the folder and ouputs results
                    """)
    parser.add_argument('--path','-p',required=False,default= None, help='Path to folder. eg. C:/data/imgs')
    parser.add_argument('--track','-t',help='track files after',
                    action="store_true")
    #Save arguments
    args = parser.parse_known_args()[0]

    #process = manager(args)
    #success = process.pipe()
    tr = Tracker(args.path)
    tr.pipe()

    print("Done")
    exit()