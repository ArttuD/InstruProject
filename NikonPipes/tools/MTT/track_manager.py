import numpy as np
from tools.MTT.imm_track import IMMTrack

class TrackManager:

    def __init__(self,dims=5,min_count=1,max_count=6,gating=10):
        self.trackers = []
        self.trackers_all = []
        self.dims = dims
        self.min_count = min_count
        self.max_count = max_count
        self.frame_count = 0
        self.gating = gating


    def update(self,dets,index,type_labels):
        # update matched trackers with assigned detections
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), self.dims))
        to_del = []
        ret = []
        #pos = []
        for t, trk in enumerate(trks):
            #print(pos[:3])
            #print('---')
            #print(self.trackers[t].imm.x)
            pos = self.trackers[t].predict()
            #print(self.trackers[t].imm.x)
            #trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        #trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = IMMTrack.associate(dets,self.trackers)
        pos = []
        for m in matched:
            #print('----')
            #print(self.trackers[m[1]].imm.x)
            self.trackers[m[1]].update(dets[m[0],np.newaxis].T)
            #print(self.trackers[m[1]].imm.x)
            self.trackers[m[1]].update_counters(True)
            self.trackers[m[1]].data_type.append(type_labels[m[0]])

        for m in unmatched_trks:
            self.trackers[m].update_counters(False)

        pos = np.array(pos)
        pos_all = []
        for k in self.trackers:
            pos_all.append(k.imm.x[:3])
        pos_all = np.array(pos_all)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if pos_all.shape[0]==0 or (np.linalg.norm(pos_all-dets[i,:3])<self.gating).sum()==0:
                trk = IMMTrack(dets[i,:],self.min_count,self.max_count)
                #print(trk.x[:3])
                trk.update_counters(True)
                self.trackers.append(trk)
                self.trackers_all.append(trk)
            else:
                print("gating")
        i = len(self.trackers)
        num_killed = 0
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.tentative:
                ret = []
                #ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.killed):
                num_killed += 1
                self.trackers.pop(i)
        if num_killed != 0:
            print("killed {} tracks".format(num_killed))
        if(len(ret)>0):
            return np.concatenate(ret)
        
        # update history
        #print('-----')
        for track in self.trackers:
            #print(track.imm.x)
            track.update_history(index)

        return np.empty((0,self.dims))