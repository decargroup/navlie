from .types import StampedValue, State, ProcessModel, Measurement
import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, x0: State, P0: np.ndarray, process_model: ProcessModel):
        self.process_model = process_model
        self.reset(x0, P0)

    def reset(self, x0: State, P0: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self._u = None  # Most recent input

    def predict(self, u: StampedValue, dt=None):
        """
        Propagates the state forward in time using a process model.
        """
        if self.x.stamp is None:
            self.x.stamp = u.stamp

        if dt is None:
            dt = u.stamp - self.x.stamp

        if self._u is not None:
            A = self.process_model.jacobian(self.x, self._u, dt)
            Q = self.process_model.covariance(self.x, self._u, dt)
            self.x = self.process_model.evaluate(self.x, self._u, dt)
            self.P = A @ self.P @ A.T + Q
            self.P = 0.5 * (self.P + self.P.T)
            self.x.stamp += dt

        self._u = u

    def correct(self, y: Measurement):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.
        """
        if self.x.stamp is None:
            self.x.stamp = y.stamp

        if y.stamp is not None:
            dt = y.stamp - self.x.stamp
            if dt > 0 and self._u is not None:
                self.predict(self._u, dt)

        R = np.atleast_2d(y.model.covariance(self.x))
        G = np.atleast_2d(y.model.jacobian(self.x))
        y_hat = y.model.evaluate(self.x)

        S = G @ self.P @ G.T + R
        K = (self.P @ G.T) @ np.linalg.inv(S)

        self.P = (np.identity(self.P.shape[0]) - K @ G) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        z = (y.value.reshape((-1, 1)) - y_hat.reshape((-1, 1)))
        dx = K @ z
        self.x.plus(dx)
