from .types import StampedValue, State, ProcessModel, Measurement
import numpy as np
from scipy.stats.distributions import chi2

class ExtendedKalmanFilter:
    def __init__(self, x0: State, P0: np.ndarray, process_model: ProcessModel):
        self.process_model = process_model
        self.reset(x0, P0)

    def reset(self, x0: State, P0: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self._u = None  

    def predict(self, u: StampedValue, dt=None, x_jac: State = None, use_last_input=True):
        """
        Propagates the state forward in time using a process model.

        Optionally, provide `x_eval` as the evaluation point for Jacobian and
        covariance functions.
        """
        if self.x.stamp is None:
            self.x.stamp = u.stamp

        if dt is None:
            dt = u.stamp - self.x.stamp

        if x_jac is None: 
            x_jac = self.x

        if use_last_input:
            u_eval = self._u 
        else: 
            u_eval = u

        if u_eval is not None:
            A = self.process_model.jacobian(x_jac, u_eval, dt)
            Q = self.process_model.covariance(x_jac, u_eval, dt)
            self.x = self.process_model.evaluate(self.x, u_eval, dt)
            self.P = A @ self.P @ A.T + Q
            self.P = 0.5 * (self.P + self.P.T)
            self.x.stamp += dt

        self._u = u

    def correct(self, y: Measurement, x_jac: State = None, reject_outlier=False):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Optionally, provide `x_eval` as the evaluation point for Jacobian and
        covariance functions.
        """
        if self.x.stamp is None:
            self.x.stamp = y.stamp

        if y.stamp is not None:
            dt = y.stamp - self.x.stamp
            if dt > 0 and self._u is not None:
                self.predict(self._u, dt)

        if x_jac is None: 
            x_jac = self.x

        R = np.atleast_2d(y.model.covariance(x_jac))
        G = np.atleast_2d(y.model.jacobian(x_jac))
        y_hat = y.model.evaluate(self.x)
        z = y.value.reshape((-1, 1)) - y_hat.reshape((-1, 1))
        S = G @ self.P @ G.T + R    

        outlier = False 

        # Test for outlier if requested.
        if reject_outlier:
            S = G @ self.P @ G.T + R  
            md = np.asscalar(z.T @ np.linalg.solve(S, z))
            if md > chi2.ppf(0.99, df=z.size):
                outlier = True

        if not outlier:
            K = np.linalg.solve(S.T, (self.P @ G.T).T).T
            #K = (self.P @ G.T) @ np.linalg.inv(S)

            self.P = (np.identity(self.P.shape[0]) - K @ G) @ self.P
            self.P = 0.5 * (self.P + self.P.T)
            dx = K @ z
            self.x.plus(dx)

# TODO: add Iterated EKF, 