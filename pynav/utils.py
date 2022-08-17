from typing import Callable, List, Tuple
from pynav.types import State
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats.distributions import chi2


class GaussianResult:
    """
    A data container that simultaneously computes various interesting metrics
    about a Gaussian filter's state estimate, given the ground-truth value of
    the state.
    """

    
    __slots__ = [
        "stamp",
        "state",
        "state_true",
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
        state_true: State = None,
    ):
        self.stamp = state.stamp
        self.state = state
        self.state_true = state_true
        self.covariance = covariance
        if state_true is not None:
            e = state.minus(state_true).reshape((-1, 1))
            cov_inv = np.linalg.inv(covariance)
            self.error = e.flatten()
            self.ees = np.asscalar(e.T @ e)
            self.nees = np.asscalar(e.T @ cov_inv @ e)
            self.md = np.sqrt(self.nees)
            self.three_sigma = 3 * np.sqrt(np.diag(covariance))


class GaussianResultList:
    """
    A data container that accepts a list of `GaussianResult` objects and
    stacks the attributes of each object numpy arrays. Convenient for plotting.
    """

    __slots__ = [
        "stamp",
        "state",
        "state_true",
        "covariance",
        "error",
        "ees",
        "nees",
        "md",
        "three_sigma",
        "value",
        "value_true",
        "dof"
    ]

    def __init__(self, result_list: List[GaussianResult]):
        self.stamp = np.array([r.stamp for r in result_list])
        self.state = np.array([r.state for r in result_list])
        self.state_true = np.array([r.state_true for r in result_list])
        self.covariance = np.array([r.covariance for r in result_list])
        self.error = np.array([r.error for r in result_list])
        self.ees = np.array([r.ees for r in result_list])
        self.nees = np.array([r.nees for r in result_list])
        self.md = np.array([r.md for r in result_list])
        self.three_sigma = np.array([r.three_sigma for r in result_list])
        self.value = np.array([r.state.value for r in result_list])
        self.dof = np.array([r.state.dof for r in result_list])
        self.value_true = np.array([r.state_true.value for r in result_list])


class MonteCarloResult:
    """
    A data container which computes various interesting metrics associated with
    Monte Carlo experiments, such as the average estimation error squared (EES)
    and the average normalized EES.
    """

    # TODO: add chi-squared bounds

    def __init__(self, trial_results: List[GaussianResultList]):
        self.num_trials = len(trial_results)
        self.stamp = trial_results[0].stamp
        self.average_nees = np.average(
            np.array([t.nees for t in trial_results]), axis=0
        )
        self.average_ees = np.average(np.array([t.ees for t in trial_results]), axis=0)

        self.rmse: np.ndarray = np.sqrt(
            np.average(np.power(np.array([t.error for t in trial_results]), 2), axis=0)
        )

        self.total_rmse: np.ndarray = np.sqrt(self.average_ees)

        self.expected_nees = np.array(trial_results[0].dof)
        self.dof = trial_results[0].dof 

    def nees_lower_bound(self, confidence_interval: float):
        if confidence_interval >= 1 or confidence_interval <= 0:
            raise ValueError("Confidence interval must lie in (0, 1)")

        lower_bound_threshold = (1-confidence_interval)/2
        return chi2.ppf(lower_bound_threshold, df=self.num_trials * self.dof)/self.num_trials

    def nees_upper_bound(self, confidence_interval: float, double_sided=True):
        if confidence_interval >= 1 or confidence_interval <= 0:
            raise ValueError("Confidence interval must lie in (0, 1)")

        
        upper_bound_threshold = confidence_interval
        if double_sided:
            upper_bound_threshold += (1-confidence_interval)/2

        return chi2.ppf(upper_bound_threshold, df=self.num_trials * self.dof)/self.num_trials
        


def monte_carlo(trial: Callable[[int], List[GaussianResult]], num_trials: int):
    """
    Monte-Carlo experiment executor. Give a callable `trial` function that
    executes a trial and returns a list of `GaussianResult`.
    """

    trial_results = [None] * num_trials
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

    return MonteCarloResult(trial_results)


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
