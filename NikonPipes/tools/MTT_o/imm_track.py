import numpy as np 
from scipy.spatial.distance import mahalanobis
from filterpy.kalman import IMMEstimator,KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import linear_sum_assignment
from tools.MTT_o.track import Track
import matplotlib.pyplot as plt


class IMMTrack(Track):
    # common observation model for each filter
    H = np.array([[1,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0]])

    def __init__(self,z,min_count,max_count,ide,gating_far):
        super().__init__(min_count,max_count,ide)
        self.gating_far = gating_far
        # previous state + additive Gaussian noise
        kf_RW = KalmanFilter(dim_x=11,dim_z=5)
        kf_RW.x = np.zeros(11)
        kf_RW.F = np.array([[1,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [1,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0]])
        kf_RW.H = IMMTrack.H
        # initial uncertainty
        kf_RW.P *= 1e2
        kf_RW.P[5:,5:] *= 10
        # measurement uncertainty (x,y,z,v,i) microm
        kf_RW.R = np.diag([1,1,1,1,1])*1e-5
        kf_RW.Q *= 1.

        # linear extrapolation of last 2
        kf_FLE = KalmanFilter(dim_x=11,dim_z=5)
        kf_FLE.x = np.zeros(11)
        kf_FLE.F = np.array([[2,0,0,0,0,-1,0,0,0,0,0],
                            [0,2,0,0,0,0,-1,0,0,0,0],
                            [0,0,2,0,0,0,0,-1,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [1,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0]])
        kf_FLE.H = IMMTrack.H
        kf_FLE.P *= 1e2
        kf_FLE.P[5:,5:] *= 10
        kf_FLE.R = np.diag([1,1,1,1,1])*1e-5
        kf_FLE.Q *= 1.

        # linear extrapolation of last 3
        kf_SLE = KalmanFilter(dim_x=11,dim_z=5)
        kf_SLE.x = np.zeros(11)
        kf_SLE.F = np.array([[3,0,0,0,0,-3,0,0,1,0,0],
                            [0,3,0,0,0,0,-3,0,0,1,0],
                            [0,0,3,0,0,0,0,-3,0,0,1],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [1,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0]])
        kf_SLE.H = IMMTrack.H
        kf_SLE.P *= 1e2
        kf_SLE.P[5:,5:] *= 10
        kf_SLE.R = np.diag([1,1,1,1,1])*1e-5
        kf_SLE.Q *= 1.

        mu = [0.33,0.33,0.33]
        trans = np.array([[0.5, 0.15,0.15],
                        [0.15, 0.7, 0.15],
                        [0.15, 0.15, 0.7]])
        filters = [kf_RW,kf_FLE,kf_SLE]
        #filters = [kf_RW,kf_RW,kf_RW]

        self.kalman = IMMEstimator(filters, mu, trans)
        # init state
        state = np.zeros(11)
        z_ = np.copy(z)
        state[:5] = z_
        state[5:8] = z_[:3]
        state[8:] = z_[:3]
        self.kalman.x = state#.reshape((-1,1))
        kf_RW.x = state
        kf_FLE.x = state
        kf_SLE.x = state

    def predict(self):
        self.kalman.predict()
        return self.kalman.x

    def update(self,z=None):
        super().update(z)
        # update IMM state
        if z is not None:
            #print(z.shape)
            self.kalman.update(z)

    def update_history(self,index):
        self.indices.append(index)
        self.history.append(np.copy(self.kalman.x[:3]))
        if self.tentative:
            self.status.append(0)
        elif self.killed:
            self.status.append(-1)
        else:
            self.status.append(1)

    def get_state(self):
        return self.kalman.x[:2]

    def gating(self,z):
        """Mahanalobis distance"""
        S = IMMTrack.H@self.kalman.P@IMMTrack.H.T+self.kalman.filters[0].R
        m = z-IMMTrack.H@self.kalman.x
        m_ = m[:,:,None]
        S_ = S[None,...]
        SI = np.linalg.inv(S_)

        return np.sqrt(m_.transpose((0,2,1))@SI@m_)[:,0,0]

    def gating_euc(self,z):
        """euclidian distance"""
        m = z-IMMTrack.H@self.kalman.x
        m_ = m[:,:,None]
        dist = np.linalg.norm(m_,axis=1)
        dist[dist>20] = np.inf
        return dist[:,0]

    @staticmethod
    def associate(measurements,trackers,gating_far):

        # calculate L
        L = np.zeros((measurements.shape[0],len(trackers)))

        # construct gating matrix
        gating = np.zeros_like(L)
        for i in range(L.shape[1]):
            gating[:,i] = trackers[i].gating(measurements)
        gating_euc = np.zeros_like(L)
        for i in range(L.shape[1]):
            gating_euc[:,i] = trackers[i].gating_euc(measurements)
        for idx,tracker in enumerate(trackers):
            likelihoods = []
            for kf in tracker.kalman.filters:
                # find the maximum likelihood among IMM filters
                # innovation covariance
                #S = kf.SI
                S = IMMTrack.H@kf.P@IMMTrack.H.T+kf.R
                # innovation
                #m = kf.y
                m = measurements-IMMTrack.H@kf.x
                # likelihood
                #lik = 1/np.linalg.det(2*np.pi*S)*np.exp(-0.5*m.T@S@m)
                m_ = m[:,:,None]
                S_ = S[None,...]
                SI = np.linalg.inv(S_)
                lik = -np.log(np.linalg.det(2*np.pi*S_))-0.5*m_.transpose((0,2,1))@SI@m_
                likelihoods.append(lik)
            #print(np.array(likelihoods).max(axis=0).shape)
            #l_i = np.max(likelihoods)
            l_i = np.array(likelihoods).max(axis=0)[:,0,0]
            L[:,idx] = l_i
            #plt.imshow(L)
            #plt.colorbar()
            #plt.show()

        # gating mahanalobis chi-squared inverse 0.05 df=5
        #print("-----")
        #print(np.mean(L))
        #L[gating>11.07] = np.inf
        if np.min(L.shape) > 0:
            L[L==-np.inf] = np.inf
            if (L==np.inf).sum()==(L.shape[0]*L.shape[1]):
                matched_indices = np.empty(shape=(0,2))
            else:
                # global optimal (paper says this is bad)
                #row_ind,col_ind = linear_sum_assignment(L)
                # assignment as described in the paper 
                mask = np.zeros_like(L)
                #mask[gating>100] = 1
                mask[gating>11.07] = np.inf
                #mask[gating>(5*11.07)] = 1
                
                # size = L.shape[0]*L.shape[1]
                # row_ind = []
                # col_ind = []
                # stop = False
                # while (mask.sum()<size) and (not stop):
                #     best_idx = np.unravel_index(np.argmax(np.ma.array(L,mask=mask)),L.shape)
                #     row_ind.append(best_idx[0])
                #     col_ind.append(best_idx[1])
                #     # mask out the entire measurement and track
                #     mask[:,best_idx[1]] = 1
                #     mask[best_idx[0],:] = 1
                
                row_ind,col_ind = linear_sum_assignment(-L)

                matched_indices = np.array(list(zip(row_ind, col_ind)))
                if len(matched_indices) == 0:
                    matched_indices = np.empty(shape=(0,2))
        else:
            matched_indices = np.empty(shape=(0,2))
        '''
        plt.subplot(1,3,1)
        plt.imshow(L)
        plt.colorbar()
        plt.subplot(1,3,2)
        g = np.zeros_like(L)
        if len(matched_indices[:,0])>0:
            g[matched_indices[:,0],matched_indices[:,1]] = 10
            plt.imshow(g)
        plt.subplot(1,3,3)
        plt.imshow(gating)
        plt.colorbar()
        plt.show()
        '''
        # gating
        unmatched_detections = []
        for d, det in enumerate(measurements):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
        #filter out matched with 
        matches = []
        for m in matched_indices:
            # gating
            if(gating_euc[m[0], m[1]]>gating_far):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)