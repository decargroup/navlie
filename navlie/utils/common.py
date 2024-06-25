""" 
Collection of miscellaneous utility functions and classes.
"""

from typing import Callable, List, Tuple, Union, Any
from joblib import Parallel, delayed
from navlie.types import State, StateWithCovariance
from navlie.lib.states import IMMState
from navlie.utils.mixture import gaussian_mixing
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import block_diag, expm
from scipy.stats.distributions import chi2

def state_interp(
    query_stamps: Union[float, List[float], Any],
    state_list: List[State],
    method="linear",
) -> Union[State, List[State]]:
    """
    Performs "linear" (geodesic) interpolation between ``State`` objects. Multiple
    interpolations can be performed at once in a vectorized fashion. If the
    query point is out of bounds, the end points are returned.

    ..code-block:: python

        x_data = [SE3State.random(stamp=i) for i in range(10)]
        x_query = [0.2, 0.5, 10]
        x_interp = state_interp(x_query, x_data)

    Parameters
    ----------
    query_stamps : float or object with ``.stamp`` attribute (or Lists thereof)
        Query stamps. Can either be a float, or an object containing a ``stamp``
        attribute. If a list is provided, it will be treated as multiple query
        points and the return value will be a list of ``State`` objects.
    state_list : List[State]
        List of ``State`` objects to interpolate between.

    Returns
    -------
    ``State`` or List[``State``]
        The interpolated state(s).

    Raises
    ------
    TypeError
        If query point is not a float or object with a ``stamp`` attribute.
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
        if not isinstance(stamp, (float, int)):
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
            (query_stamps[inside] - stamp_lower[inside])
            / (stamp_upper[inside] - stamp_lower[inside])
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
        self.rmse = np.sqrt(self.ees / state.dof)
        #:float: Mahalanobis distance
        self.md = np.sqrt(self.nees)
        #:numpy.ndarray: three-sigma bounds on each error component
        self.three_sigma = 3 * np.sqrt(np.diag(covariance))


class GaussianResultList:
    """
    A data container that accepts a list of ``GaussianResult`` objects and
    stacks the attributes in numpy arrays. Convenient for plotting. This object
    does nothing more than array-ifying the attributes of ``GaussianResult``.

    This object also supports slicing, which will return a new ``GaussianResultList``
    object with the sliced attributes either through time or through the degrees
    of freedom themselves. For example,

    .. code-block:: python

        results = GaussianResultList.from_estimates(estimates, state_true_list)

        results[0:10] # returns the first 10 time steps
        results[:, 0] # returns the first degree of freedom
        results[0:10, 0] # returns the first degree of freedom for the first 10 time steps
        results[0:10, [0, 1]] # returns the first two degrees of freedom for the first 10 time steps
        results[:, 3:] # returns all degrees of freedom except the first three

    This can be very useful if you want to examine the nees or rmse of just some
    states, or if you want to plot the error of just some states. For example,
    if you have an SE3State, where the first 3 degrees of freedom are attitude,
    and the last 3 are position, you can plot only the attitude error with

    .. code-block:: python

        nav.plot_error(results[:, 0:3])

    and likewise get only the attitude NEES with ``results[:, 0:3].nees``.

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


        Let ``N = len(result_list)``
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
        if isinstance(key, tuple):
            if not len(key) == 2:
                raise IndexError("Only two dimensional indexing is supported")
        else:
            key = (key, slice(None, None, None))

        key_lists = list(key)  # make mutable
        for i, k in enumerate(key):
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
        out.covariance = self.covariance[
            np.ix_(key_lists[0], key_lists[1], key_lists[1])
        ]  # (N, key_size, key_size)
        out.error = self.error[key1, key2]  # (N, key_size)
        out.ees = np.sum(np.atleast_2d(out.error**2), axis=1)

        if len(out.error.shape) == 1:
            out.nees = out.error**2 / out.covariance.flatten()
            out.dof = np.ones_like(out.stamp)
        else:
            out.nees = np.sum(
                out.error * np.linalg.solve(out.covariance, out.error), axis=1
            )
            out.dof = out.error.shape[1] * np.ones_like(out.stamp)

        out.md = np.sqrt(out.nees)
        out.three_sigma = 3 * np.sqrt(
            np.diagonal(out.covariance, axis1=1, axis2=2)
        )
        out.rmse = np.sqrt(out.ees / out.dof)
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


        Let ``N`` denote the number of time steps in a trial.
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


class IMMResult(GaussianResult):
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
        "model_probabilities",
    ]

    def __init__(self, imm_estimate: IMMState, state_true: State):
        super().__init__(
            gaussian_mixing(
                imm_estimate.model_probabilities, imm_estimate.model_states
            ),
            state_true,
        )

        self.model_probabilities = imm_estimate.model_probabilities


class IMMResultList(GaussianResultList):
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
        "model_probabilities",
    ]

    def __init__(self, result_list: List[IMMResult]):
        super().__init__(result_list)
        self.model_probabilities = np.array(
            [r.model_probabilities for r in result_list]
        )

    @staticmethod
    def from_estimates(
        estimate_list: List[IMMState],
        state_true_list: List[State],
        method="nearest",
    ):
        """
        A convenience function that creates a IMMResultList from a list of
        IMMState and a list of true State objects

        Parameters
        ----------
        estimate_list : List[IMMState]
            A list of IMMState objects
        state_true_list : List[State]
            A list of true State objects
        method : "nearest" or "linear", optional
            The method used to interpolate the true state when the timestamps
            do not line up exactly, by default "nearest".

        Returns
        -------
        IMMResultList
            A IMMResultList object
        """
        stamps = [r.model_states[0].stamp for r in estimate_list]

        state_true_list = state_interp(stamps, state_true_list, method=method)
        return IMMResultList(
            [
                IMMResult(estimate, state_true)
                for estimate, state_true in zip(estimate_list, state_true_list)
            ]
        )


def monte_carlo(
    trial: Callable[[int], GaussianResultList],
    num_trials: int,
    num_jobs: int = -1,
    verbose: int = 10,
) -> MonteCarloResult:
    """
    Monte-Carlo experiment executor. Give a callable ``trial`` function that
    executes a trial and returns a ``GaussianResultList``, and this function
    will execute it a number of times and aappgregate the results.

    Parameters
    ----------
    trial : Callable[[int], GaussianResultList]
        Callable trial function. Must accept a single integer trial number,
        and return a GaussianResultList. From trial to trial, the timestamps
        are expected to remain consistent.
    num_trials : int
        Number of Trials to execute
    num_jobs: int, optional
        The maximum number of concurrently running jobs, by default -1.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. For num_jobs below -1,
        (n_cpus + 1 + num_jobs) are used. Thus for num_jobs = -2, all CPUs but
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
    trial_results = Parallel(n_jobs=num_jobs, verbose=verbose)(
        delayed(trial)(i) for i in range(num_trials)
    )

    return MonteCarloResult(trial_results)


def randvec(cov: np.ndarray, num_samples: int = 1) -> np.ndarray:
    """

    Produces a random zero-mean column vector with covariance given by ``cov``

    Parameters
    ----------
    cov : np.ndarray
        square numpy array with shape (n,n)
    num_samples : int, optional
        Will make ``num_samples`` independent random vectors and
        stack them horizontally, by default 1. It can be faster to generate
        many samples this way to avoid recomputing the Cholesky decomposition
        every time.

    Returns
    -------
    np.ndarray with shape (n, num_samples)
        Random column vector(s) with covariance ``cov``

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

    where :math:``\mathbf{Q}_c`` is a power spectral density,
    Van Loan's method can be used to find its equivalent discrete-time representation,

    .. math::
        \mathbf{x}_k = \mathbf{A}_{d} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \hspace{5mm}
        \mathbf{w} \sim \mathcal{N} (\mathbf{0}, \mathbf{Q}_d ).

    These are computed using the matrix exponential, with a sampling period :math:``\Delta t``.

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
    offset_step = 1 / freq
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


def find_nearest_stamp_idx(
    stamps_list: List[float], stamp: Union[float, List[float]]
) -> int:
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

    This function is also compatible with ``State`` objects, and hence
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
        if method == "cs":
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

    func_to_diff = lambda dx: output_diff(
        fun(input_plus(x, dx), *args, **kwargs), Y_bar
    )

    # Check if input/output has a dof attribute. otherwise, assume it will
    # behave like a numpy array and use the ``.size`` attribute to get
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
            jac_fd[:, i] = (Y_plus - Y_minus).ravel() / (2 * step_size)

        elif method == "cs":
            Y_imag: State = func_to_diff(1j * dx.copy())
            jac_fd[:, i] = np.imag(Y_imag).ravel() / step_size

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Must be 'forward', 'central' or 'cs"
            )

    return jac_fd


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
        self.rmse = np.sqrt(self.ees / state.dof)
        #:float: Mahalanobis distance
        self.md = np.sqrt(self.nees)
        #:numpy.ndarray: three-sigma bounds on each error component
        self.three_sigma = 3 * np.sqrt(np.diag(covariance))


class GaussianResultList:
    """
    A data container that accepts a list of ``GaussianResult`` objects and
    stacks the attributes in numpy arrays. Convenient for plotting. This object
    does nothing more than array-ifying the attributes of ``GaussianResult``.

    This object also supports slicing, which will return a new ``GaussianResultList``
    object with the sliced attributes either through time or through the degrees
    of freedom themselves. For example,

    .. code-block:: python

        results = GaussianResultList.from_estimates(estimates, state_true_list)

        results[0:10] # returns the first 10 time steps
        results[:, 0] # returns the first degree of freedom
        results[0:10, 0] # returns the first degree of freedom for the first 10 time steps
        results[0:10, [0, 1]] # returns the first two degrees of freedom for the first 10 time steps
        results[:, 3:] # returns all degrees of freedom except the first three

    This can be very useful if you want to examine the nees or rmse of just some
    states, or if you want to plot the error of just some states. For example,
    if you have an SE3State, where the first 3 degrees of freedom are attitude,
    and the last 3 are position, you can plot only the attitude error with

    .. code-block:: python

        nav.plot_error(results[:, 0:3])

    and likewise get only the attitude NEES with ``results[:, 0:3].nees``.

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


        Let ``N = len(result_list)``
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
        if isinstance(key, tuple):
            if not len(key) == 2:
                raise IndexError("Only two dimensional indexing is supported")
        else:
            key = (key, slice(None, None, None))

        key_lists = list(key)  # make mutable
        for i, k in enumerate(key):
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
        out.covariance = self.covariance[
            np.ix_(key_lists[0], key_lists[1], key_lists[1])
        ]  # (N, key_size, key_size)
        out.error = self.error[key1, key2]  # (N, key_size)
        out.ees = np.sum(np.atleast_2d(out.error**2), axis=1)

        if len(out.error.shape) == 1:
            out.nees = out.error**2 / out.covariance.flatten()
            out.dof = np.ones_like(out.stamp)
        else:
            out.nees = np.sum(
                out.error * np.linalg.solve(out.covariance, out.error), axis=1
            )
            out.dof = out.error.shape[1] * np.ones_like(out.stamp)

        out.md = np.sqrt(out.nees)
        out.three_sigma = 3 * np.sqrt(
            np.diagonal(out.covariance, axis1=1, axis2=2)
        )
        out.rmse = np.sqrt(out.ees / out.dof)
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


        Let ``N`` denote the number of time steps in a trial.
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


class IMMResult(GaussianResult):
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
        "model_probabilities",
    ]

    def __init__(self, imm_estimate: IMMState, state_true: State):
        super().__init__(
            gaussian_mixing(
                imm_estimate.model_probabilities, imm_estimate.model_states
            ),
            state_true,
        )

        self.model_probabilities = imm_estimate.model_probabilities


class IMMResultList(GaussianResultList):
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
        "model_probabilities",
    ]

    def __init__(self, result_list: List[IMMResult]):
        super().__init__(result_list)
        self.model_probabilities = np.array(
            [r.model_probabilities for r in result_list]
        )

    @staticmethod
    def from_estimates(
        estimate_list: List[IMMState],
        state_true_list: List[State],
        method="nearest",
    ):
        """
        A convenience function that creates a IMMResultList from a list of
        IMMState and a list of true State objects

        Parameters
        ----------
        estimate_list : List[IMMState]
            A list of IMMState objects
        state_true_list : List[State]
            A list of true State objects
        method : "nearest" or "linear", optional
            The method used to interpolate the true state when the timestamps
            do not line up exactly, by default "nearest".

        Returns
        -------
        IMMResultList
            A IMMResultList object
        """
        stamps = [r.model_states[0].stamp for r in estimate_list]

        state_true_list = state_interp(stamps, state_true_list, method=method)
        return IMMResultList(
            [
                IMMResult(estimate, state_true)
                for estimate, state_true in zip(estimate_list, state_true_list)
            ]
        )
