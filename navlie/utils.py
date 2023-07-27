from typing import Callable, List, Tuple, Union, Any, Dict
from joblib import Parallel, delayed
from navlie.types import State, Measurement, StateWithCovariance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from scipy.interpolate import interp1d
from scipy.linalg import block_diag, expm


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
        "rmse",
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

        e = state_true.minus(state).reshape((-1, 1))
        #:numpy.ndarray: error vector between estimated and true state
        self.error = e.ravel()
        #:float: sum of estimation error squared (EES)
        self.ees = np.ndarray.item(e.T @ e)
        #:float: normalized estimation error squared (NEES)
        self.nees = np.ndarray.item(e.T @ np.linalg.solve(covariance, e))
        #:float: root mean squared error (RMSE)
        self.rmse = np.sqrt(self.ees/state.dof)
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
        "rmse",
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
        #:numpy.ndarray with shape (N,): EES throughout trajectory
        self.rmse = np.array([r.rmse for r in result_list])
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

    def __getitem__(self, key):
        # TODO need more tests for all cases!
        if isinstance(key, tuple):
            if not len(key) == 2:
                raise IndexError("Only two dimensional indexing is supported")
        else: 
            key = (key, slice(None, None, None))

        key_lists = list(key) # make mutable
        for i,k in enumerate(key):
            if isinstance(k, int):
                key_lists[i] = [k]
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else self.error.shape[i]
                step = k.step if k.step is not None else 1
                key_lists[i] = list(range(start, stop, step))
            elif isinstance(k, list):
                pass 
            else:
                raise TypeError("keys must be int, slice, or list of indices")
        

        key1, key2 = key
        out = GaussianResultList([])
        out.stamp = self.stamp[key1]
        out.state = self.state[key1]
        out.state_true = self.state_true[key1]
        out.covariance = self.covariance[np.ix_(key_lists[0], key_lists[1], key_lists[1])] # (N, key_size, key_size)
        out.error = self.error[key1,key2] # (N, key_size)
        out.ees = np.sum(np.atleast_2d(out.error**2), axis=1)

        if len(out.error.shape) == 1:
            out.nees = out.error**2 / out.covariance.flatten()
            out.dof = np.ones_like(out.stamp)
        else:
            out.nees = np.sum(out.error * np.linalg.solve(out.covariance, out.error), axis=1)
            out.dof = out.error.shape[1] * np.ones_like(out.stamp)
            
        out.md = np.sqrt(out.nees)
        out.three_sigma = 3 * np.sqrt(np.diagonal(out.covariance, axis1=1, axis2=2))
        out.rmse = np.sqrt(out.ees/out.dof)
        out.value = self.value[key1]
        out.value_true = self.value_true[key1] 
        out.covariance = out.covariance.squeeze()
        out.error = out.error.squeeze()

        return out



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


        An example of how to make a NEES plot with both upper and lower bounds:

        .. code-block:: python

            ax.plot(results.stamp, results.nees)
            ax.plot(results.stamp, results.nees_lower_bound(0.99))
            ax.plot(results.stamp, results.nees_upper_bound(0.99))
        """
        if confidence_interval >= 1 or confidence_interval <= 0:
            raise ValueError("Confidence interval must lie in (0, 1)")

        lower_bound_threshold = (1 - confidence_interval) / 2
        return chi2.ppf(lower_bound_threshold, df=self.dof)

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

        An example of how to make a NEES plot with only upper bounds:

        .. code-block:: python

            ax.plot(results.stamp, results.nees)
            ax.plot(results.stamp, results.nees_upper_bound(0.99, double_sided=False))

        """
        if confidence_interval >= 1 or confidence_interval <= 0:
            raise ValueError("Confidence interval must lie in (0, 1)")

        upper_bound_threshold = confidence_interval
        if double_sided:
            upper_bound_threshold += (1 - confidence_interval) / 2

        return chi2.ppf(upper_bound_threshold, df=self.dof)

    @staticmethod
    def from_estimates(
        estimate_list: List[StateWithCovariance],
        state_true_list: List[State],
        method="nearest",
    ):
        """
        A convenience function that creates a GaussianResultList from a list of
        StateWithCovariance and a list of true State objects

        Parameters
        ----------
        estimate_list : List[StateWithCovariance]
            A list of StateWithCovariance objects
        state_true_list : List[State]
            A list of true State objects
        method : "nearest" or "linear", optional
            The method used to interpolate the true state when the timestamps
            do not line up exactly, by default "nearest".

        Returns
        -------
        GaussianResultList
            A GaussianResultList object
        """
        stamps = [r.stamp for r in estimate_list]

        state_true_list = state_interp(stamps, state_true_list, method=method)
        return GaussianResultList(
            [
                GaussianResult(estimate, state_true)
                for estimate, state_true in zip(estimate_list, state_true_list)
            ]
        )


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
        self.average_nees: np.ndarray = np.average(
            np.array([t.nees for t in trial_results]), axis=0
        )
        self.nees = self.average_nees
        #:numpy.ndarray with shape (N,): average EES throughout trajectory
        self.average_ees: np.ndarray = np.average(
            np.array([t.ees for t in trial_results]), axis=0
        )
        self.ees = self.average_ees
        #:numpy.ndarray with shape (N,dof): root-mean-squared error of each component
        self.rmse: np.ndarray = np.sqrt(
            np.average(
                np.power(np.array([t.error for t in trial_results]), 2), axis=0
            )
        )
        #:numpy.ndarray with shape (N,): Total RMSE, this can be meaningless if units differ in a state
        self.total_rmse: np.ndarray = np.sqrt(self.average_ees)
        #:numpy.ndarray with shape (N,1): expected NEES value throughout trajectory
        self.expected_nees: np.ndarray = np.array(trial_results[0].dof)
        #:numpy.ndarray with shape (N): dof throughout trajectory
        self.dof: np.ndarray = trial_results[0].dof

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
    trial: Callable[[int], GaussianResultList],
    num_trials: int,
    n_jobs: int = -1,
    verbose: int = 10,
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
    n_jobs: int, optional
        The maximum number of concurrently running jobs, by default -1.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
        one are used.
    verbose: int, optional
        The verbosity level, by default 10. If non zero, progress messages
        are printed. Above 50, the output is sent to stdout. The frequency
        of the messages increases with the verbosity level. If it more than
        10, all iterations are reported.

    Returns
    -------
    MonteCarloResult
        Data container object
    """
    trial_results = [None] * num_trials

    print("Starting Monte Carlo experiment...")
    trial_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(trial)(i) for i in range(num_trials)
    )

    return MonteCarloResult(trial_results)


def randvec(cov: np.ndarray, num_samples: int = 1) -> np.ndarray:
    """

    Produces a random zero-mean column vector with covariance given by `cov`

    Parameters
    ----------
    cov : np.ndarray
        square numpy array with shape (n,n)
    num_samples : int, optional
        Will make `num_samples` independent random vectors and
        stack them horizontally, by default 1. It can be faster to generate
        many samples this way to avoid recomputing the Cholesky decomposition
        every time.

    Returns
    -------
    np.ndarray with shape (n, num_samples)
        Random column vector(s) with covariance `cov`

    """
    L = np.linalg.cholesky(cov)
    return L @ np.random.normal(0, 1, (cov.shape[0], num_samples))


def van_loans(
    A_c: np.ndarray,
    L_c: np.ndarray,
    Q_c: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Van Loan's method for computing the discrete-time A and Q matrices.

    Given a continuous-time system of the form

    .. math::
        \dot{\mathbf{x}} = \mathbf{A}_c \mathbf{x} + \mathbf{L}_c \mathbf{w}, \hspace{5mm}
        \mathbf{w} \sim \mathcal{N} (\mathbf{0}, \mathbf{Q}_c ),

    where :math:`\mathbf{Q}_c` is a power spectral density,
    Van Loan's method can be used to find its equivalent discrete-time representation,

    .. math::
        \mathbf{x}_k = \mathbf{A}_{d} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \hspace{5mm}
        \mathbf{w} \sim \mathcal{N} (\mathbf{0}, \mathbf{Q}_d ).

    These are computed using the matrix exponential, with a sampling period :math:`\Delta t`.

    Parameters
    ----------
    A_c : np.ndarray
        Continuous-time A matrix.
    L_c : np.ndarray
        Continuous-time L matrix.
    Q_c : np.ndarray
        Continuous-time noise matrix
    dt : float
        Discretization timestep.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A_d and Q_d, discrete-time matrices.
    """
    N = A_c.shape[0]

    A_c = np.atleast_2d(A_c)
    L_c = np.atleast_2d(L_c)
    Q_c = np.atleast_2d(Q_c)

    # Form Xi matrix and compute Upsilon using matrix exponential
    Xi = block_diag(A_c, -A_c.T, A_c, np.zeros((N, N)))
    Xi[:N, N : 2 * N] = L_c @ Q_c @ L_c.T
    Upsilon = expm(Xi * dt)

    # Extract relevant parts of Upsilon
    A_d = Upsilon[:N, :N]
    Q_d = Upsilon[:N, N : 2 * N] @ A_d.T

    return A_d, Q_d


def plot_error(
    results: GaussianResultList,
    axs: List[plt.Axes] = None,
    label: str = None,
    sharey: bool = False,
    color=None,
    bounds=True,
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

    if dim < 3:
        n_rows = dim
    else:
        n_rows = 3

    n_cols = int(np.ceil(dim / 3))

    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=sharey)
    else:
        fig: plt.Figure = axs.ravel("F")[0].get_figure()

    axs_og = axs
    kwargs = {}
    if color is not None:
        kwargs["color"] = color

    axs: List[plt.Axes] = axs.ravel("F")
    for i in range(results.three_sigma.shape[1]):
        if bounds:
            axs[i].fill_between(
                results.stamp,
                results.three_sigma[:, i],
                -results.three_sigma[:, i],
                alpha=0.5,
                **kwargs,
            )
        axs[i].plot(results.stamp, results.error[:, i], label=label, **kwargs)


    fig: plt.Figure = fig # For type hinting
    return fig, axs_og

def plot_nees(
    results: GaussianResultList,
    axs: List[plt.Axes] = None,
    label: str = None,
    color=None,
    confidence_interval: float = 0.95,
    normalize: bool = False,
    expected_nees_color = "r",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Makes a plot of the NEES, showing the actual NEES values, the expected NEES,
    and the bounds of the specified confidence interval.

    Parameters
    ----------
    results : GaussianResultList or MonteCarloResult
        Results to plot
    axs : List[plt.Axes], optional
        Axes on which to draw, by default None. If None, new axes will be
        created.
    label : str, optional
        Label to assign to the NEES line, by default None
    color : optional
        Fed directly to the ``plot(..., color=color)`` function, by default None
    confidence_interval : float or None, optional
        Desired probability confidence region, by default 0.95. Must lie between
        0 and 1. If None, no confidence interval will be plotted.
    normalize : bool, optional
        Whether to normalize the NEES by the degrees of freedom, by default False

    Returns
    -------
    plt.Figure
        Figure on which the plot was drawn  
    plt.Axes
        Axes on which the plot was drawn
    """

    if axs is None:
        fig, axs = plt.subplots(1, 1, sharex=True,)
    else:
        fig = axs.get_figure()

    axs_og = axs
    kwargs = {}
    if color is not None:
        kwargs["color"] = color

    if normalize:
        s = results.dof
    else:
        s = 1


    expected_nees_label = "Expected NEES"
    ci_label = f"${int(confidence_interval*100)}\%$ CI"
    _, exisiting_labels = axs.get_legend_handles_labels()

    if expected_nees_label in exisiting_labels:
        expected_nees_label = None
    if ci_label in exisiting_labels:
        ci_label = None

    # fmt:off
    axs.plot(results.stamp, results.nees/s, label=label, **kwargs)
    if confidence_interval:
        axs.plot(results.stamp, results.dof/s, label=expected_nees_label, color=expected_nees_color)
        axs.plot(results.stamp, results.nees_upper_bound(confidence_interval)/s, "--", color="k", label=ci_label)
        axs.plot(results.stamp, results.nees_lower_bound(confidence_interval)/s, "--", color="k")
    # fmt:on

    axs.legend()

    return fig, axs_og

def plot_meas(
    meas_list: List[Measurement],
    state_list: List[State],
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
    state_list : List[State]
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

    meas_list.sort(key=lambda y: y.stamp)

    # Find the state of the nearest timestamp to the measurement
    y_stamps = np.array([y.stamp for y in meas_list])
    x_stamps = np.array([x.stamp for x in state_list])
    indexes = np.array(range(len(state_list)))
    nearest_state = interp1d(
        x_stamps,
        indexes,
        "nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    state_idx = nearest_state(y_stamps)
    y_meas = []
    y_true = []
    three_sigma = []
    for i in range(len(meas_list)):
        data = np.ravel(meas_list[i].value)
        y_meas.append(data)
        x = state_list[int(state_idx[i])]
        y = meas_list[i].model.evaluate(x)
        if y is None:
            y_true.append(np.zeros_like(data)*np.nan)
            three_sigma.append(np.zeros_like(data)*np.nan)
        else:
            y_true.append(np.ravel(y))
            R = np.atleast_2d(meas_list[i].model.covariance(x))
            three_sigma.append(3 * np.sqrt(np.diag(R)))

    y_meas = np.atleast_2d(np.array(y_meas))
    y_true = np.atleast_2d(np.array(y_true))
    three_sigma = np.atleast_2d(np.array(three_sigma))
    y_stamps = np.array(y_stamps)  # why is this necessary?
    x_stamps = np.array(x_stamps)

    # Plot
    size_y = np.size(meas_list[0].value)
    if axs is None:
        fig, axs = plt.subplots(size_y, 1, sharex=True, sharey=sharey)
    else:
        if isinstance(axs, plt.Axes):
            axs = np.array([axs])

        fig = axs.ravel("F")[0].get_figure()
    axs = np.atleast_1d(axs)
    for i in range(size_y):
        axs[i].scatter(
            y_stamps, y_meas[:, i], color="b", alpha=0.7, s=2, label="Measured"
        )
        axs[i].plot(
            y_stamps, y_true[:, i], color="r", alpha=1, label="Modelled"
        )
        axs[i].fill_between(
            y_stamps,
            y_true[:, i] + three_sigma[:, i],
            y_true[:, i] - three_sigma[:, i],
            alpha=0.5,
            color="r",
        )
    return fig, axs


def plot_meas_by_model(
    meas_list: List[Measurement],
    state_list: List[State],
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
    state_list : List[State]
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

    # Create sub-lists for every model ID
    meas_by_model: Dict[int, List[Measurement]] = {}
    for meas in meas_list:
        model_id = id(meas.model)
        if model_id not in meas_by_model:
            meas_by_model[model_id] = []
        meas_by_model[model_id].append(meas)

    if axs is None:
        axs = np.array([None]*len(meas_by_model))

    axs = axs.ravel("F")

    figs = [None]*len(meas_by_model)
    for i, temp in enumerate(meas_by_model.items()):
        model_id, meas_list = temp
        fig, ax = plot_meas(meas_list, state_list, axs[i], sharey=sharey)
        figs[i] = fig
        axs[i] = ax
        ax[0].set_title(f"{meas_list[0].model} {hex(model_id)}", fontsize=12)
        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].tick_params(axis='both', which='minor', labelsize=8)
    

    return fig, axs
    


def plot_poses(
    poses,
    ax: plt.Axes = None,
    line_color: str = None,
    triad_color: str = None,
    arrow_length: float = 1,
    step: int = 5,
    label: str = None,
):
    """
    Plots position trajectory in 3D
    and poses along the trajectory as triads.

    Parameters
    ----------
    poses : List[SE3State]
        A list objects containing a `position` property (numpy array of size 3)
        and an `attitude` (3 x 3 numpy array) property representing the rotation 
        matrix :math:`\mathbf{C}_{ab}`.
    ax : plt.Axes, optional
        Axes to plot on, if none, 3D axes are created.
    line_color : str, optional
        Color of the position trajectory.
    triad_color : str, optional
        Triad color. If none are specified, defaults to RGB.
    arrow_length : int, optional
        Triad arrow length, by default 1.
    step : int or None, optional
        Step size in list of poses, by default 5. If None, no triads are plotted.
    label : str, optional
        Optional label for the triad
    """
    # TODO. handle 2D case
    if isinstance(poses, GaussianResultList):
        poses = poses.state

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
    else:
        fig = ax.get_figure()

    if triad_color is None:
        colors = ["tab:red", "tab:green", "tab:blue"]  # Default to RGB
    else:
        colors = [triad_color] * 3

    # Plot a line for the positions
    r = np.array([pose.position for pose in poses])
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2], color=line_color, label=label)

    # Plot triads using quiver
    if step is not None:
        C = np.array([poses[i].attitude.T for i in range(0, len(poses), step)])
        r = np.array([poses[i].position for i in range(0, len(poses), step)])
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        ax.quiver(
            x,
            y,
            z,
            C[:, 0, 0],
            C[:, 0, 1],
            C[:, 0, 2],
            color=colors[0],
            length=arrow_length,
            arrow_length_ratio=0.1,
        )
        ax.quiver(
            x,
            y,
            z,
            C[:, 1, 0],
            C[:, 1, 1],
            C[:, 1, 2],
            color=colors[1],
            length=arrow_length,
            arrow_length_ratio=0.1,
        )
        ax.quiver(
            x,
            y,
            z,
            C[:, 2, 0],
            C[:, 2, 1],
            C[:, 2, 2],
            color=colors[2],
            length=arrow_length,
            arrow_length_ratio=0.1,
        )

    set_axes_equal(ax)
    return fig, ax


def set_axes_equal(ax: plt.Axes):
    """
    Sets the axes of a 3D plot to have equal scale.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    length = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - length, x_middle + length])
    ax.set_ylim3d([y_middle - length, y_middle + length])
    ax.set_zlim3d([z_middle - length, z_middle + length])


def state_interp(
    query_stamps: Union[float, List[float], Any],
    state_list: List[State],
    method="linear",
) -> Union[State, List[State]]:
    """
    Performs "linear" (geodesic) interpolation between `State` objects. Multiple
    interpolations can be performed at once in a vectorized fashion. If the
    query point is out of bounds, the end points are returned.

    ..code-block:: python

        x_data = [SE3State.random(stamp=i) for i in range(10)]
        x_query = [0.2, 0.5, 10]
        x_interp = state_interp(x_query, x_data)

    Parameters
    ----------
    query_stamps : float or object with `.stamp` attribute (or Lists thereof)
        Query stamps. Can either be a float, or an object containing a `stamp`
        attribute. If a list is provided, it will be treated as multiple query
        points and the return value will be a list of `State` objects.
    state_list : List[State]
        List of `State` objects to interpolate between.

    Returns
    -------
    `State` or List[`State`]
        The interpolated state(s).

    Raises
    ------
    TypeError
        If query point is not a float or object with a `stamp` attribute.
    """
    
    # Handle input
    if isinstance(query_stamps, list):
        single_query = False
    elif isinstance(query_stamps, np.ndarray):
        single_query = False
    elif isinstance(query_stamps, float):
        query_stamps = [query_stamps]
        single_query = True
    else:
        pass    


    # if not isinstance(query_stamps, list):
    #     query_stamps = [query_stamps]
    #     single_query = True
    # else:
    #     single_query = False

    query_stamps = query_stamps.copy()
    for i, stamp in enumerate(query_stamps):
        if not isinstance(stamp, float):
            if hasattr(stamp, "stamp"):
                stamp = stamp.stamp
                query_stamps[i] = stamp
            else:
                raise TypeError(
                    "Stamps must be of type float or have a stamp attribute"
                )

    # Get the indices of the states just before and just after.
    query_stamps = np.array(query_stamps)
    state_list = np.array(state_list)
    stamp_list = [state.stamp for state in state_list]
    stamp_list.sort()
    stamp_list = np.array(stamp_list)
    if method == "linear":
        idx_middle = np.interp(
            query_stamps, stamp_list, np.array(range(len(stamp_list)))
        )
        idx_lower = np.floor(idx_middle).astype(int)
        idx_upper = idx_lower + 1

        before_start = query_stamps < stamp_list[0]
        after_end = idx_upper >= len(state_list)
        inside = np.logical_not(np.logical_or(before_start, after_end))

        # Return endpoint if out of bounds
        idx_upper[idx_upper == len(state_list)] = len(state_list) - 1

        # ############ Do the interpolation #################
        stamp_lower = stamp_list[idx_lower]
        stamp_upper = stamp_list[idx_upper]

        # "Fraction" of the way between the two states
        alpha = np.zeros(len(query_stamps))
        alpha[inside] = np.array(
            (query_stamps[inside] - stamp_lower[inside]) / (stamp_upper[inside] - stamp_lower[inside])
        ).ravel()

        # The two neighboring states around the query point
        state_lower: List[State] = np.array(state_list[idx_lower]).ravel()
        state_upper: List[State] = np.array(state_list[idx_upper]).ravel()

        # Interpolate between the two states
        dx = np.array(
            [s.minus(state_lower[i]).ravel() for i, s in enumerate(state_upper)]
        )

        out = []
        for i, state in enumerate(state_lower):
            if np.isnan(alpha[i]) or np.isinf(alpha[i]) or alpha[i] < 0.0:
                raise RuntimeError("wtf")
            
            state_interp = state.plus(dx[i] * alpha[i])

            state_interp.stamp = query_stamps[i]
            out.append(state_interp)

    elif method == "nearest":

        indexes = np.array(range(len(stamp_list)))
        nearest_state = interp1d(
            stamp_list,
            indexes,
            "nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        state_idx = nearest_state(query_stamps).astype(int)
        out = state_list[state_idx].tolist()

    if single_query:
        out = out[0]

    return out

def schedule_sequential_measurements(model_list, freq):
    """Schedules sequential measurements from a list of MeasurementModels
    that cannot generate measurements at the same time. This allows
    looping through the measurement model list one at a time. 

    Parameters
    ----------
    model_list: List[MeasurementModel]
        The list of sequential MeasurementModels.
    freq: float
        The overall frequency in which all the measurements are generated.

    Returns
    -------
    List[float]
        The list of initial offsets associated with each MeasurementModel.
    float
        The reduced frequency at which each individual MeasurementModel
        generates measurements.
    """
    n_models = len(model_list)
    offset_list = [None] * n_models
    offset_step = (1 / freq)
    new_freq = freq / n_models
    
    for i in range(n_models):
        offset_list[i] = i * offset_step

    return offset_list, new_freq 

def associate_stamps(
    first_stamps: List[float],
    second_stamps: List[float],
    offset: float = 0.0,
    max_difference: float = 0.02,
) -> List[Tuple[int, int]]:
    """
    Associate timestamps.

    Returns a sorted list of matches, of length of the smallest of
    first_stamps and second_stamps.

    Function taken from rpg_trajectory_evaluation toolbox.

    Parameters
    ----------
    first_stamps : List[float]
        List of first timestamps
    second_stamps : List[float]
        List of second timestamps
    offset : float, optional
        Offset between the two lists, by default 0.0.
    max_difference : float, optional
        Maximum difference between stamps in the two list
        to be considered a match, by default 0.02.

    Returns
    -------
    List[Tuple[int, int]]
        Sorted list of matches in the form (match_first_idx, match_second_idx).
    """
    potential_matches = [
        (abs(a - (b + offset)), idx_a, idx_b)
        for idx_a, a in enumerate(first_stamps)
        for idx_b, b in enumerate(second_stamps)
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()  # prefer the closest
    matches = []
    first_idxes = list(range(len(first_stamps)))
    second_idxes = list(range(len(second_stamps)))
    for diff, idx_a, idx_b in potential_matches:
        if idx_a in first_idxes and idx_b in second_idxes:
            first_idxes.remove(idx_a)
            second_idxes.remove(idx_b)
            matches.append((int(idx_a), int(idx_b)))

    matches.sort()
    return matches


def find_nearest_stamp_idx(stamps_list: List[float], stamp: Union[float, List[float]]) -> int:
    """
    Find the index of the nearest stamp in ``stamps_list`` to ``stamp``. If
    ``stamp`` is a list or array, then the output is a list of indices.

    Parameters
    ----------
    stamps_list : List[float]
        List of timestamps
    stamp : float or List[float] or numpy.ndarray
        Query stamp(s).

    Returns
    -------
    int or List[int]
        Index of nearest stamp.
    """

    if isinstance(stamp, float):
        single_query = True
        query_stamps = np.array([stamp])
    else:
        single_query = False
        query_stamps = np.array(stamp).ravel()


    nearest_stamp = interp1d(
        stamps_list,
        np.array(range(len(stamps_list))),
        "nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )

    out = nearest_stamp(query_stamps).astype(int).tolist()

    if single_query:
        out = out[0]

    return out


def jacobian(
    fun: Callable,
    x: Union[np.ndarray, State],
    step_size=None,
    method="forward",
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Compute the Jacobian of a function. Example use:

    .. code-block:: python

        x = np.array([1, 2]).reshape((-1,1))
        
        A = np.array([[1, 2], [3, 4]])
        def fun(x):
            return 1/np.sqrt(x.T @ A.T @ A @ x)

        jac_test = jacobian(fun, x, method=method)
        jac_true = (- x.T @ A.T @ A)/((x.T @ A.T @ A @ x)**(3/2))

        assert np.allclose(jac_test, jac_true, atol=1e-6)

    This function is also compatible with `State` objects, and hence
    can compute on-manifold derivatives. Example use:

    .. code-block:: python

        T = SE23State([0.1,0.2,0.3,4,5,6,7,8,9], direction="right")

        def fun(T: SE23State):
            # Returns the normalized body-frame velocity vector
            C_ab = T.attitude
            v_zw_a = T.velocity
            v_zw_b = C_ab.T @ v_zw_a
            return v_zw_b/np.linalg.norm(v_zw_b)

        jac_fd = jacobian(fun, T)

    Parameters
    ----------
    fun : Callable
        function to compute the Jacobian of
    x : Union[np.ndarray, State]
        input to the function
    step_size : float, optional
        finite difference step size, by default 1e-6
    method : str, optional
        "forward", "central" or "cs", by default "forward". "forward" calculates
        using a forward finite difference procedure. "central" calculates using
        a central finite difference procedure. "cs" calculates using the
        complex-step procedure. If using "cs", you must be careful to ensure 
        that the function can handle and propagate through complex 
        components.

    Returns
    -------
    np.ndarray with shape (M, N)
        Jacobian of the function, where ``M`` is the DOF of the output and
        ``N`` is the DOF of the input.
    """
    x = x.copy() 

    if step_size is None:
        if method=="cs":
            step_size = 1e-16
        else:
            step_size = 1e-6

    # Check if input has a plus method. otherwise, assume it will behave
    # like a numpy array
    if hasattr(x, "plus"):
        input_plus = lambda x, dx: x.plus(dx)
    else:
        input_plus = lambda x, dx: x + dx.reshape(x.shape)

    Y_bar: State = fun(x.copy(), *args, **kwargs)

    # Check if output has a minus method. otherwise, assume it will behave
    # like a numpy array 
    if hasattr(Y_bar, "minus"):
        output_diff = lambda Y, Y_bar: Y.minus(Y_bar)
    else:
        output_diff = lambda Y, Y_bar: Y - Y_bar



    func_to_diff = lambda dx : output_diff(fun(input_plus(x, dx), *args, **kwargs), Y_bar)

    # Check if input/output has a dof attribute. otherwise, assume it will
    # behave like a numpy array and use the `.size` attribute to get
    # the DOF of the input/output
    if hasattr(x, "dof"):
        N = x.dof
    else:
        N = x.size

    if hasattr(Y_bar, "dof"):
        M = Y_bar.dof
    else:
        M = Y_bar.size


    Y_bar_diff = func_to_diff(np.zeros((N,)))
    jac_fd = np.zeros((M, N))

    # Main loop to calculate jacobian
    for i in range(N):
        dx = np.zeros((N))
        dx[i] = step_size

        if method == "forward":
            Y_plus: State = func_to_diff(dx.copy())
            jac_fd[:, i] = (Y_plus - Y_bar_diff).ravel() / step_size

        elif method == "central":
            Y_plus = func_to_diff(dx.copy())
            Y_minus = func_to_diff(-dx.copy())
            jac_fd[:, i] = (Y_plus - Y_minus).ravel() / (2*step_size)

        elif method == "cs":
            Y_imag: State = func_to_diff(1j*dx.copy())
            jac_fd[:, i] = np.imag(Y_imag).ravel() / step_size

        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Must be 'forward', 'central' or 'cs")


    return jac_fd
