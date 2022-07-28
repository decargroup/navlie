from typing import Callable, List, Tuple
from pynav.types import State
import numpy as np
import matplotlib.pyplot as plt
import time

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

    def __init__(
        self,
        state: State,
        covariance: np.ndarray = None,
        state_gt: State = None,
    ):
        self.stamp = state.stamp
        self.state = state
        self.state_gt = state_gt
        self.covariance = covariance
        if state_gt is not None:
            e = state.minus(state_gt).reshape((-1, 1))
            cov_inv = np.linalg.inv(covariance)
            self.error = e.flatten()
            self.ees = np.asscalar(e.T @ e)
            self.nees = np.asscalar(e.T @ cov_inv @ e)
            self.md = np.sqrt(self.nees)
            self.three_sigma = 3 * np.sqrt(np.diag(covariance))


class GaussianResultList:
    """
    A data container that accepts a list of `GaussianResult` objects and
    stacks the attributes of each object into new lists. Convenient for plotting.
    """

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
    # TODO: just write this explicitly..
    

    def __init__(self, result_list: List[GaussianResult]):
        props = GaussianResult.__slots__
        for p in props:
            data_list = [getattr(r, p) for r in result_list]
            try:
                setattr(self, p, np.array(data_list))
            except:
                setattr(self, p, data_list)


class MonteCarloResults:
    """
    A data container which computes various interesting metrics associated with
    Monte Carlo experiments, such as the average estimation error squared (EES)
    and the average normalized EES.
    """
    # TODO: add chi-squared bounds, expected NEES
    # Should we support, or check, for time stamps that arnt aligned?
    # probably not since MonteCarlo is only ever in simulation, where timestamps
    # will almost always be the same from trial-to-trial.

    def __init__(self, trial_results: List[GaussianResultList]):
        self.num_trials = len(trial_results)
        self.stamp = trial_results[0].stamp
        self.average_nees = np.average(
            np.array([t.nees for t in trial_results]), axis=0
        )
        self.average_ees = np.average(
            np.array([t.ees for t in trial_results]), axis=0
        )

        self.rmse: np.ndarray = np.sqrt(np.average(
            np.power(np.array([t.error for t in trial_results]),2), axis=0
        ))

        self.total_rmse: np.ndarray  = np.sqrt(self.average_ees)



def montecarlo(trial: Callable[[int], List[GaussianResult]], num_trials: int):
    """
    Monte-Carlo experiment executor. Give a callable `trial` function that
    executes a trial and returns a list of `GaussianResult`.
    """

    trial_results = [None]*num_trials
    print("Starting Monte Carlo experiment...")
    for i in range(num_trials):
        print("Trial {0} of {1}... ".format(i + 1, num_trials))
        start_time = time.time()

        # Execute the trial
        trial_results[i] = GaussianResultList(trial(i))

        # Print some info
        duration = time.time() - start_time
        remaining = (num_trials - i - 1) * duration
        print(
            "    Completed in {duration:.1f}s. Estimated time remaining: {remaining:.1f}s".format(
                duration=duration, remaining=remaining
            )
        )

    return MonteCarloResults(trial_results)


def randvec(cov: np.ndarray):
    """
    Produces a random zero-mean column vector with covariance given by `cov`
    """
    return np.linalg.cholesky(cov) @ np.random.normal(0, 1, (cov.shape[0], 1))


def plot_error(
    results: GaussianResultList,
    axs: List[plt.Axes] = None,
    label=None,
    sharey=False,
    color=None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    A generic three-sigma bound plotter.

    """
    dim = results.error.shape[1]

    n_rows = 3
    n_cols = int(np.ceil(dim / 3))

    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=sharey)
    else:
        fig = axs.ravel("F")[0].get_figure()

    axs_og = axs
    kwargs = {}
    if color is not None:
        kwargs["color"] = color

    
    axs: List[plt.Axes] = axs.ravel("F")
    for i in range(len(axs)):
        axs[i].fill_between(
            results.stamp,
            results.three_sigma[:, i],
            -results.three_sigma[:, i],
            alpha=0.5,
            **kwargs,
        )
        axs[i].plot(results.stamp, results.error[:, i], label=label, **kwargs)

    return fig, axs_og
