import numpy as np
from tools.cMTT.imm_track import IMMTrack
from tools.cMTT.velo_track import VeloTrack
from tools.cMTT.imm_simple import IMMSimple 

class TrackManager:

    def __init__(self,dims=5,min_count=5,max_count=5,gating=50, gating_ = 500):
        self.trackers = []
        self.trackers_all = []
        self.dims = dims
        self.min_count = min_count
        self.max_count = max_count
        self.frame_count = 0
        self.gating_ = gating_
        self.gating = gating
        self.models = [IMMTrack,VeloTrack,IMMSimple]
        self.ide = 0


    def update(self,dets,index):
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
        matched, unmatched_dets, unmatched_trks = VeloTrack.associate(dets,self.trackers, self.gating_)
        pos = []
        for m in matched:
            #print('----')
            #print(self.trackers[m[1]].imm.x)
            self.trackers[m[1]].update(dets[m[0],np.newaxis].T)
            #print(self.trackers[m[1]].imm.x)
            self.trackers[m[1]].update_counters(True)

        for m in unmatched_trks:
            self.trackers[m].update_counters(False)

        pos = np.array(pos)
        pos_all = []
        for k in self.trackers:
            pos_all.append(k.kalman.x[:3])
        pos_all = np.array(pos_all)
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            #if pos_all.shape[0]==0 or (np.linalg.norm(pos_all-dets[i,:3])<self.gating).sum()==0:
            if pos_all.shape[0]==0 or (np.linalg.norm(pos_all-dets[i,:3],axis=-1)<self.gating).sum()==0:
                trk = VeloTrack(dets[i,:],self.min_count,self.max_count,self.ide)
                self.ide +=1
                #print(trk.x[:3])
                trk.update_counters(True)
                self.trackers.append(trk)
                self.trackers_all.append(trk)
            else:
                #print("gating")
                pass
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