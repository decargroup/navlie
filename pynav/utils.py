from pynav.types import State
import numpy as np


class GaussianResults:
    __slots__ = [
        "stamp",
        "state",
        "state_gt",
        "covariance",
        "error",
        "ees",
        "nees",
        "md",
        "three_sigma",
    ]

    def __init__(self, state: State, covariance: np.ndarray, state_gt: State):
        self.stamp = state.stamp
        self.state = state
        self.state_gt = state_gt
        self.covariance = covariance

        e = state.minus(state_gt).reshape((-1, 1))
        self.error = e.flatten()
        self.ees = np.asscalar(e.T @ e)
        self.nees = np.asscalar(e.T @ np.linalg.solve(covariance, e))
        self.md = np.sqrt(self.nees)
        self.three_sigma = 3 * np.sqrt(np.diag(covariance))
