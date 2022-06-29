from typing import Callable, List
from pynav.types import State
import numpy as np


class GaussianResult:
    """
    A data container that simultaneously computes various interesting metrics
    about a Gaussian filter's state estimate, given the ground-truth value of 
    the state.
    """

    # All properties MUST be added to __slots__
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

    def __init__(self, state: State, covariance: np.ndarray = None, state_gt: State = None):
        self.stamp = state.stamp
        self.state = state
        self.state_gt = state_gt
        self.covariance = covariance

        if state_gt is not None:
            e = state.minus(state_gt).reshape((-1, 1))
            self.error = e.flatten()
            self.ees = np.asscalar(e.T @ e)
            self.nees = np.asscalar(e.T @ np.linalg.solve(covariance, e))
            self.md = np.sqrt(self.nees)
            self.three_sigma = 3 * np.sqrt(np.diag(covariance))

class GaussianResultList:
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
    def __init__(self, result_list: List[GaussianResult]):
        props = GaussianResult.__slots__ 
        for p in props:
            data_list = [getattr(r, p) for r in result_list]
            try:
                setattr(self, p, np.array(data_list))
            except:
                setattr(self, p, data_list)



class MonteCarloResults:
    def __init__(self, trial_results: List[GaussianResultList]):
        self.num_trials = len(trial_results)
        self.stamp = trial_results[0].stamp
        self.average_nees = np.average(
            np.array([t.nees for t in trial_results]), axis=0
            )
        self.average_ees = np.average(
            np.array([t.ees for t in trial_results]), axis=0
            )




def montecarlo(trial: Callable[[None], List[GaussianResult]], num_trials):
    trial_results = []
    print("Starting Monte Carlo experiment...")
    for i in range(num_trials):
        trial_results.append(GaussianResultList(trial()))
        print("Trial {0} of {1}.".format(i+1, num_trials))

    return MonteCarloResults(trial_results)