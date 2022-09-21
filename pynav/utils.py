from typing import Callable, List, Tuple
from pynav.types import State, Measurement, StateWithCovariance
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats.distributions import chi2
from scipy.interpolate import interp1d
from tqdm import tqdm


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
        estimate: StateWithCovariance,
        state_true: State,
    ):
        """
        Parameters
        ----------
        estimate : StateWithCovariance
            Estimated state and corresponding covariance.
        state_true : State
            The true state, which will be used to compute various error metrics.
        """

        state = estimate.state
        covariance = estimate.covariance

        #:float: timestamp
        self.stamp = state.stamp
        #:State: estimated state
        self.state = state
        #:State: true state
        self.state_true = state_true
        #:numpy.ndarray: covariance associated with estimated state
        self.covariance = covariance

        e = state.minus(state_true).reshape((-1, 1))
        cov_inv = np.linalg.inv(covariance)
        #:numpy.ndarray: error vector between estimated and true state
        self.error = e.flatten()
        #:float: sum of estimation error squared (EES)
        self.ees = np.ndarray.item(e.T @ e)
        #:float: normalized estimation error squared (NEES)
        self.nees = np.ndarray.item(e.T @ cov_inv @ e)
        #:float: Mahalanobis distance
        self.md = np.sqrt(self.nees)
        #:numpy.ndarray: three-sigma bounds on each error component
        self.three_sigma = 3 * np.sqrt(np.diag(covariance))


class GaussianResultList:
    """
    A data container that accepts a list of `GaussianResult` objects and
    stacks the attributes in numpy arrays. Convenient for plotting. This object
    does nothing more than array-ifying the attributes of `GaussianResult`
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
        "dof",
    ]

    def __init__(self, result_list: List[GaussianResult]):
        """
        Parameters
        ----------
        result_list : List[GaussianResult]
            A list of GaussianResult, intended such that each element corresponds
            to a different time point


        Let `N = len(result_list)`
        """
        #:numpy.ndarray with shape (N,):  timestamp
        self.stamp = np.array([r.stamp for r in result_list])
        #:numpy.ndarray with shape (N,): numpy array of State objects
        self.state: List[State] = np.array([r.state for r in result_list])
        #:numpy.ndarray with shape (N,): numpy array of true State objects
        self.state_true: List[State] = np.array(
            [r.state_true for r in result_list]
        )
        #:numpy.ndarray with shape (N,dof,dof): covariance
        self.covariance: np.ndarray = np.array(
            [r.covariance for r in result_list]
        )
        #:numpy.ndarray with shape (N, dof): error throughout trajectory
        self.error = np.array([r.error for r in result_list])
        #:numpy.ndarray with shape (N,): EES throughout trajectory
        self.ees = np.array([r.ees for r in result_list])
        #:numpy.ndarray with shape (N,): NEES throughout trajectory
        self.nees = np.array([r.nees for r in result_list])
        #:numpy.ndarray with shape (N,): Mahalanobis distance throughout trajectory
        self.md = np.array([r.md for r in result_list])
        #:numpy.ndarray with shape (N, dof): three-sigma bounds
        self.three_sigma = np.array([r.three_sigma for r in result_list])
        #:numpy.ndarray with shape (N,): state value. type depends on implementation
        self.value = np.array([r.state.value for r in result_list])
        #:numpy.ndarray with shape (N,): dof throughout trajectory
        self.dof = np.array([r.state.dof for r in result_list])
        #:numpy.ndarray with shape (N,): true state value. type depends on implementation
        self.value_true = np.array([r.state_true.value for r in result_list])


class MonteCarloResult:
    """
    A data container which computes various interesting metrics associated with
    Monte Carlo experiments, such as the average estimation error squared (EES)
    and the average normalized EES.
    """

    def __init__(self, trial_results: List[GaussianResultList]):
        """
        Parameters
        ----------
        trial_results : List[GaussianResultList]
            Each GaussianResultList corresponds to a trial. This object assumes
            that the timestamps in each trial are identical.


        Let `N` denote the number of time steps in a trial.
        """

        #:List[GaussianResultList]: raw trial results
        self.trial_results = trial_results
        #:int: number of trials
        self.num_trials = len(trial_results)
        #:numpy.ndarray with shape (N,): timestamps throughout trajectory
        self.stamp = trial_results[0].stamp
        #:numpy.ndarray with shape (N,): average NEES throughout trajectory
        self.average_nees = np.average(
            np.array([t.nees for t in trial_results]), axis=0
        )
        #:numpy.ndarray with shape (N,): average EES throughout trajectory
        self.average_ees = np.average(
            np.array([t.ees for t in trial_results]), axis=0
        )
        #:numpy.ndarray with shape (N,dof): root-mean-squared error of each component
        self.rmse: np.ndarray = np.sqrt(
            np.average(
                np.power(np.array([t.error for t in trial_results]), 2), axis=0
            )
        )
        #:numpy.ndarray with shape (N,): Total RMSE, this can be meaningless if units differ in a state
        self.total_rmse: np.ndarray = np.sqrt(self.average_ees)
        #:numpy.ndarray with shape (N,1): expected NEES value throughout trajectory
        self.expected_nees = np.array(trial_results[0].dof)
        #:numpy.ndarray with shape (N): dof throughout trajectory
        self.dof = trial_results[0].dof

    def nees_lower_bound(self, confidence_interval: float):
        """
        Calculates the NEES lower bound throughout the trajectory.

        Parameters
        ----------
        confidence_interval : float
            Single-sided cumulative probability threshold that defines the bound.
            Must be between 0 and 1

        Returns
        -------
        numpy.ndarray with shape (N,)
            NEES value corresponding to confidence interval
        """
        if confidence_interval >= 1 or confidence_interval <= 0:
            raise ValueError("Confidence interval must lie in (0, 1)")

        lower_bound_threshold = (1 - confidence_interval) / 2
        return (
            chi2.ppf(lower_bound_threshold, df=self.num_trials * self.dof)
            / self.num_trials
        )

    def nees_upper_bound(self, confidence_interval: float, double_sided=True):
        """
        Calculates the NEES upper bound throughout the trajectory

        Parameters
        ----------
        confidence_interval : float
            Cumulative probability threshold that defines the bound. Must be
            between 0 and 1.
        double_sided : bool, optional
            Whether the provided threshold is single-sided or double sided,
            by default True

        Returns
        -------
        numpy.ndarray with shape (N,)
            NEES value corresponding to confidence interval

        """
        if confidence_interval >= 1 or confidence_interval <= 0:
            raise ValueError("Confidence interval must lie in (0, 1)")

        upper_bound_threshold = confidence_interval
        if double_sided:
            upper_bound_threshold += (1 - confidence_interval) / 2

        return (
            chi2.ppf(upper_bound_threshold, df=self.num_trials * self.dof)
            / self.num_trials
        )


def monte_carlo(
    trial: Callable[[int], GaussianResultList], num_trials: int
) -> MonteCarloResult:
    """
    Monte-Carlo experiment executor. Give a callable `trial` function that
    executes a trial and returns a `GaussianResultList`, and this function
    will execute it a number of times and aappgregate the results.

    Parameters
    ----------
    trial : Callable[[int], GaussianResultList]
        Callable trial function. Must accept a single integer trial number,
        and return a GaussianResultList. From trial to trial, the timestamps
        are expected to remain consistent.
    num_trials : int
        Number of Trials to execute

    Returns
    -------
    MonteCarloResult
        Data container object
    """
    trial_results = [None] * num_trials

    print("Starting Monte Carlo experiment...")
    for i in tqdm(range(num_trials), unit="trial", ncols=80):
        # Execute the trial
        trial_results[i] = trial(i)

    return MonteCarloResult(trial_results)


def randvec(cov: np.ndarray) -> np.ndarray:
    """
    Produces a random zero-mean column vector with covariance given by `cov`
    """
    return np.linalg.cholesky(cov) @ np.random.normal(0, 1, (cov.shape[0], 1))


def plot_error(
    results: GaussianResultList,
    axs: List[plt.Axes] = None,
    label: str = None,
    sharey: bool = False,
    color=None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    A generic three-sigma bound plotter.

    Parameters
    ----------
    results : GaussianResultList
        Contains the data to plot
    axs : List[plt.Axes], optional
        Axes to draw on, by default None. If None, new axes will be created.
    label : str, optional
        Text label to add, by default None
    sharey : bool, optional
        Whether to have a common y axis or not, by default False
    color : color, optional
        specify the color of the error/bounds.

    Returns
    -------
    plt.Figure
        Handle to figure.
    List[plt.Axes]
        Handle to axes that were drawn on.
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


def plot_meas(
    meas_list: List[Measurement],
    state_true_list: List[State],
    axs: List[plt.Axes] = None,
    sharey=False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Given measurement data, make time-domain plots of the measurement values
    and their ground-truth model-based values.

    Parameters
    ----------
    meas_list : List[Measurement]
        Measurement data to be plotted.
    state_true_list : List[State]
        A list of true State objects with similar timestamp domain. Will be
        interpolated if timestamps do not line up perfectly.
    axs : List[plt.Axes], optional
        Axes to draw on, by default None. If None, new axes will be created.
    sharey : bool, optional
        Whether to have a common y axis or not, by default False

    Returns
    -------
    plt.Figure
        Handle to figure.
    List[plt.Axes]
        Handle to axes that were drawn on.
    """

    # Convert everything to numpy arrays for plotting, and compute the
    # ground-truth model-based measurement value.
    y_stamps = np.array([y.stamp for y in meas_list])
    x_stamps = np.array([x.stamp for x in state_true_list])
    nearest_state = interp1d(
        x_stamps,
        np.array(range(len(state_true_list))),
        "nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    state_idx = nearest_state(y_stamps)
    y_meas = []
    y_true = []
    for i in range(len(meas_list)):
        y_meas.append(np.ravel(meas_list[i].value))
        x = state_true_list[int(state_idx[i])]
        y_true.append(np.ravel(meas_list[i].model.evaluate(x)))

    y_meas = np.atleast_2d(np.array(y_meas))
    y_true = np.atleast_2d(np.array(y_true))
    y_stamps = np.array(y_stamps)
    x_stamps = np.array(x_stamps)

    # Plot

    size_y = np.size(meas_list[0].value)
    if axs is None:
        fig, axs = plt.subplots(size_y, 1, sharex=True, sharey=sharey)
    else:
        if isinstance(axs, plt.Axes):
            axs = np.array([axs])

        fig = axs.ravel("F")[0].get_figure()

    for i in range(size_y):
        axs[i].scatter(
            y_stamps, y_meas[:, i], color="b", alpha=0.7, s=2, label="Measured"
        )
        axs[i].plot(
            y_stamps, y_true[:, i], color="r", alpha=1, label="Modelled"
        )

    return fig, axs
