""" 
Module containing many predict-correct style filters.
"""

from typing import List
from navlie.types import (
    Input,
    State,
    ProcessModel,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from tqdm import tqdm
from filters import ExtendedKalmanFilter
from scipy.stats import multivariate_normal
from navlie.utils import GaussianResultList, GaussianResult, state_interp
from .imm import gaussian_mixing

class GMMState:
    __slots__ = ["states", "weights"]

    def __init__(
        self,
        states: List[StateWithCovariance],
        weights: List[float],
    ):
        self.states = states
        self.weights = weights

    @property
    def stamp(self):
        return self.states[0].state.stamp

    def copy(self) -> "GMMState":
        x_copy = [x.copy() for x in self.states]
        return GMMState(x_copy, self.weights.copy())

class GSFResult(GaussianResult):
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
        "weights",
    ]

    def __init__(self, gsf_estimate: GMMState, state_true: State):
        super().__init__(
            gaussian_mixing(
                gsf_estimate.weights, gsf_estimate.states
            ),
            state_true,
        )

        self.weights = gsf_estimate.weights


class GaussianSumFilter:
    """
    On-manifold Gaussian Sum Filter.
    """

    __slots__ = [
        "process_model",
        "reject_outliers",
    ]

    def __init__(
        self,
        process_model: ProcessModel,
        reject_outliers=False,
    ):
        """
        Parameters
        ----------
        process_models : List[ProcessModel]
            process models to be used in the prediction step
        reject_outliers : bool, optional
            whether to apply the NIS test to measurements, by default False
        """
        self.ekf = ExtendedKalmanFilter(process_model, reject_outliers)

    def predict(
        self,
        x: GMMState,
        u: Input,
        dt: float = None,
    ) -> GMMState:
        """
        Propagates the state forward in time using a process model. The user
        must provide the current state, input, and time interval

        .. note::
            If the time interval ``dt`` is not provided in the arguments, it will
            be taken as the difference between the input stamp and the state stamp.

        Parameters
        ----------
        x : GMMState
            The current states and their associated weights.
        u : Input
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, dt will be calculated
            with ``dt = u.stamp - x.state.stamp``.
        Returns
        -------
        GMMState
            New predicted states with associated weights.
        """

        n_modes = len(x.states)

        x_check = []
        for i in range(n_modes):
            x_check.append(self.ekf.predict(x.states[i], u, dt))
        return GMMState(x_check, x.weights)
    
    def correct(
        self,
        x: GMMState,
        u: Input,
        y: Measurement,
    ) -> GMMState:
        """
        Corrects the state estimate using a measurement. The user must provide
        the current state and measurement.

        Parameters
        ----------
        x : GMMState
            The current states and their associated weights.
        y : Measurement
            Measurement to correct the state estimate.

        Returns
        -------
        GMMState
            Corrected states with associated weights.
        """
        n_modes = len(x.states)
        weights_check = x.weights.copy()

        x_hat = []
        weights_hat = []
        for i in range(n_modes):
            x, details_dict = self.ekf.correct(x.states[i], y, u)
            x_hat.append(x)
            z = details_dict["z"]
            S = details_dict["S"]
            model_likelihood = multivariate_normal.pdf(
                z.ravel(), mean=np.zeros(z.shape), cov=S
            )
            weights_hat.append(weights_check[i] * model_likelihood)

        # If all model likelihoods are zero to machine tolerance, np.sum(mu_k)=0 and it fails
        # Add this fudge factor to get through those cases.
        if np.allclose(weights_hat, np.zeros(weights_hat.shape)):
            weights_hat = 1e-10 * np.ones(weights_hat.shape)

        weights_hat = np.array(weights_hat) / np.sum(weights_hat)
            
        return GMMState(x_hat, weights_hat)    


def run_gsf_filter(
    filter: GaussianSumFilter,
    x0: State,
    P0: np.ndarray,
    input_data: List[Input],
    meas_data: List[Measurement],
    disable_progress_bar: bool = False,
) -> List[StateWithCovariance]:
    """
    Executes a predict-correct-style filter given lists of input and measurement
    data.

    Parameters
    ----------
    filter : GaussianSumFilter
        _description_
    x0 : State
        _description_
    P0 : np.ndarray
        _description_
    input_data : List[Input]
        _description_
    meas_data : List[Measurement]
        _description_
    """
    x = StateWithCovariance(x0, P0)
    if x.state.stamp is None:
        raise ValueError("x0 must have a valid timestamp.")

    # Sort the data by time
    input_data.sort(key=lambda x: x.stamp)
    meas_data.sort(key=lambda x: x.stamp)

    # Remove all that are before the current time
    for idx, u in enumerate(input_data):
        if u.stamp >= x.state.stamp:
            input_data = input_data[idx:]
            break

    for idx, y in enumerate(meas_data):
        if y.stamp >= x.state.stamp:
            meas_data = meas_data[idx:]
            break

    meas_idx = 0
    if len(meas_data) > 0:
        y = meas_data[meas_idx]

    n_modes = 1
    weights = [1.0]
    x = GMMState(
        [StateWithCovariance(x0, P0)] * n_modes, weights
    )

    results_list = []
    for k in tqdm(range(len(input_data) - 1), disable=disable_progress_bar):
        u = input_data[k]
        # Fuse any measurements that have occurred.
        if len(meas_data) > 0:
            while y.stamp < input_data[k + 1].stamp and meas_idx < len(
                meas_data
            ):
                x = filter.correct(x, y, u)
                meas_idx += 1
                if meas_idx < len(meas_data):
                    y = meas_data[meas_idx]

        results_list.append(x)
        dt = input_data[k + 1].stamp - x.stamp
        x = filter.predict(x, u, dt)

    return results_list