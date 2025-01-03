import numpy as np 
from scipy.spatial.distance import mahalanobis
from filterpy.kalman import IMMEstimator,KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import linear_sum_assignment
from tools.MTT_o.track import Track
import matplotlib.pyplot as plt


class IMMSimple(Track):
    # common observation model for each filter
    H = np.array([[1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,1]])

    def __init__(self,z,min_count,max_count,ide,gating_far):
        super().__init__(min_count,max_count,ide)
        self.gating_far = gating_far
        # previous state + additive Gaussian noise
        velo = KalmanFilter(dim_x=8,dim_z=5)
        velo.x = np.zeros(8)
        velo.F = np.array([[1,0,0,1,0,0,0,0],
                            [0,1,0,0,1,0,0,0],
                            [0,0,1,0,0,1,0,0],
                            [0,0,0,1,0,0,0,0],
                            [0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,1]])
        velo.H = IMMSimple.H
        # initial uncertainty
        velo.P *= 1e2
        #self.kalman.P[5:,5:] *= 10
        # measurement uncertainty (x,y,z,v,i) microm
        velo.R = np.diag([0.1,0.1,1,10,10])*10
        from filterpy.common import Q_discrete_white_noise
        Q = Q_discrete_white_noise(dim=2, dt=0.1, var=1.,block_size=3,order_by_dim=False)
        from scipy.linalg import block_diag
        velo.Q = block_diag(Q,np.eye(2))

        noise = KalmanFilter(dim_x=8,dim_z=5)
        noise.x = np.zeros(8)
        noise.F = np.array([[1,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0],
                            [0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,1]])
        noise.H = IMMSimple.H
        # initial uncertainty
        noise.P *= 1e2
        #self.kalman.P[5:,5:] *= 10
        # measurement uncertainty (x,y,z,v,i) microm
        noise.R = np.diag([0.1,0.1,1,10,10])*10
        noise.Q *= 1.0 
        mu = [0.2,0.8]
        trans = np.array([[0.95, 0.05],
                        [0.05, 0.95]])
        filters = [velo,noise]
        self.kalman = IMMEstimator(filters, mu, trans)
        state = np.zeros(8)
        z_ = np.copy(z)
        state[:3] = z_[:3]
        state[-2:] = z_[-2:]
        self.kalman.x = state

    def predict(self):
        self.kalman.predict()
        return self.kalman.x

    def update(self,z=None):
        super().update(z)
        # update kalman state
        if z is not None:
            #print(z.shape)
            self.kalman.update(z)
            val = self.kalman.x[3:6]
            val = np.minimum(np.abs(val),10)*np.sign(val)
            self.kalman.x[3:6] = val

    def update_history(self,index):
        self.indices.append(index)
        self.history.append(np.copy(self.kalman.x[:3]))
        if self.tentative:
            self.status.append(0)
        elif self.killed:
            self.status.append(-1)
        elif self.missed>0:
            self.status.append(2)
        else:
            self.status.append(1)

    def get_state(self):
        return self.kalman.x[:2]

    def gating(self,z,idx):
        """Mahanalobis distance"""
        S = IMMSimple.H@self.kalman.P@IMMSimple.H.T+self.kalman.filters[idx].R
        m = z-IMMSimple.H@self.kalman.x
        m_ = m[:,:,None]
        S_ = S[None,...]
        SI = np.linalg.inv(S_+np.eye(S_.shape[0])*1e-3)

        return np.sqrt(m_.transpose((0,2,1))@SI@m_)[:,0,0]

    def gating_euc(self,z):
        """euclidian distance"""
        m = z-IMMSimple.H@self.kalman.x
        m_ = m[:,:,None]
        dist = np.linalg.norm(m_,axis=1)
        dist[dist>50] = np.inf
        return dist[:,0]

    @staticmethod
    def associate(measurements,trackers,gating_far):

        # calculate L
        L = np.zeros((measurements.shape[0],len(trackers)))

        # construct gating matrix
        gating = np.zeros_like(L)
        gating2 = np.zeros_like(L)
        for i in range(L.shape[1]):
            gating[:,i] = trackers[i].gating(measurements,0)
            gating2[:,i] = trackers[i].gating(measurements,1)
        gating_euc = np.zeros_like(L)
        for i in range(L.shape[1]):
            gating_euc[:,i] = trackers[i].gating_euc(measurements)
        for idx,tracker in enumerate(trackers):
            likelihoods = []
            for kf in tracker.kalman.filters:
                # find the maximum likelihood among IMM filters
                # innovation covariance
                #S = kf.SI
                S = IMMSimple.H@kf.P@IMMSimple.H.T+kf.R
                # innovation
                #m = kf.y
                m = measurements-IMMSimple.H@kf.x
                # likelihood
                #lik = 1/np.linalg.det(2*np.pi*S)*np.exp(-0.5*m.T@S@m)
                m_ = m[:,:,None]
                S_ = S[None,...]
                SI = np.linalg.inv(S_+np.eye(S_.shape[0])*1e-3)
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
                mask[gating2>11.07] = np.inf
                #mask[gating>(5*11.07)] = 1
                
                size = L.shape[0]*L.shape[1]
                row_ind = []
                col_ind = []
                stop = False
                while (mask.sum()<size) and (not stop):
                    best_idx = np.unravel_index(np.argmax(np.ma.array(L,mask=mask)),L.shape)
                    row_ind.append(best_idx[0])
                    col_ind.append(best_idx[1])
                    # mask out the entire measurement and track
                    mask[:,best_idx[1]] = 1
                    mask[best_idx[0],:] = 1
                
                #row_ind,col_ind = linear_sum_assignment(-L)

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