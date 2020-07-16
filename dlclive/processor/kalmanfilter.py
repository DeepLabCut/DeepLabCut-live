"""
DeepLabCut-live Toolbox (deeplabcut.org)
Please see AUTHORS for contributors.
Licensed under GNU Lesser General Public License v3.0
"""


import time
import numpy as np
from dlclive.processor import Processor


class KalmanFilterPredictor(Processor):


    def __init__(self,
                 adapt=True,
                 forward=0.002,
                 fps=30,
                 nderiv=2,
                 priors=[10, 1],
                 initial_var=5,
                 process_var=5,
                 dlc_var=10,
                 **kwargs):

        super().__init__(**kwargs)

        self.adapt=adapt
        self.forward = forward
        self.dt = 1.0 / fps
        self.nderiv = nderiv
        self.priors = np.hstack(([1e5], priors))
        self.initial_var = initial_var
        self.process_var = process_var
        self.dlc_var = dlc_var
        self.is_initialized = False
        self.last_pose_time = 0


    def _get_forward_model(self, dt):

        F = np.zeros((self.n_states, self.n_states))
        for d in range(self.nderiv+1):
            for i in range(self.n_states - (d * self.bp * 2)):
                F[i, i + (2 * self.bp * d)] = (dt ** d) / max(1, d)

        return F

    
    def _init_kf(self, pose):

        # get number of body parts
        self.bp = pose.shape[0]
        self.n_states = self.bp * 2 * (self.nderiv+1)

        # initialize state matrix, set position to first pose
        self.X = np.zeros((self.n_states, 1))
        self.X[:(self.bp * 2)] = pose[:, :2].reshape(self.bp * 2, 1)

        # initialize covariance matrix, measurement noise and process noise
        self.P = np.eye(self.n_states) * self.initial_var
        self.R = np.eye(self.n_states) * self.dlc_var
        self.Q = np.eye(self.n_states) * self.process_var

        # initialize forward model:
        self.F = self._get_forward_model(self.dt)

        self.H = np.eye(self.n_states)
        self.K = np.zeros((self.n_states, self.n_states))
        self.I = np.eye(self.n_states)

        # initialize priors for forward prediction step only
        B = np.repeat(self.priors, self.bp * 2)
        self.B = B.reshape(B.size, 1)

        self.is_initialized = True


    def _predict(self):

        #self.F = self._get_forward_model(time.time()-self.last_pose_time)
        self.Xp = np.dot(self.F, self.X)
        self.Pp = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    
    def _get_residuals(self, pose):

        z = np.zeros((self.n_states, 1))
        z[:(self.bp * 2)] = pose[:self.bp, :2].reshape(self.bp * 2, 1)
        for i in range(self.bp * 2, self.n_states):
            z[i] = (z[i - (self.bp * 2)] - self.X[i - (self.bp * 2)]) / self.dt
        self.y = z - np.dot(self.H, self.Xp)


    def _update(self):

        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.X = self.Xp + np.dot(K, self.y)
        self.P = np.dot(self.I - np.dot(K, self.H), self.Pp)


    def _get_future_pose(self, dt):

        Ff = self._get_forward_model(dt)

        Pf = np.diag(self.P).reshape(self.P.shape[0], 1)
        Xf = (1 / ((1 / Pf) + (1 / self.B))) * (self.X / Pf)
        Xfp = np.dot(Ff, Xf)

        future_pose = Xfp[:(self.bp * 2)].reshape(self.bp, 2)

        return future_pose


    def _get_state_likelihood(self, pose):

        liks = pose[:, 2]
        liks_xy = np.repeat(liks, 2)
        liks_xy_deriv = np.tile(liks_xy, self.nderiv)
        liks_state = liks_xy_deriv.reshape(liks_xy_deriv.shape[0], 1)
        return(liks_state)
            
        
    def process(self, pose, **kwargs):

        if not self.is_initialized:

            self._init_kf(pose)

            self.last_pose_time = time.time()
            return pose
        
        else:
            
            self._predict()
            self._get_residuals(pose)
            self._update()

            forward_time = (time.time() - kwargs['frame_time'] + self.forward) if self.adapt else self.forward
            future_pose = self._get_future_pose(forward_time)
            future_pose = np.hstack((future_pose, pose[:,2].reshape(self.bp,1)))
            
            self.last_pose_time = time.time()
            return future_pose
