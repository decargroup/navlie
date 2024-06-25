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
from navlie.lib.states import MixtureState
import numpy as np
from tqdm import tqdm
from navlie.filters import ExtendedKalmanFilter
from scipy.stats import multivariate_normal
from navlie.utils import GaussianResult, GaussianResultList, gaussian_mixing


class GaussianSumFilter:
    """
    On-manifold Gaussian Sum Filter.
    """

    __slots__ = [
        "process_model",
        "reject_outliers",
        "ekf",
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
        x: MixtureState,
        u: Input,
        dt: float = None,
    ) -> MixtureState:
        """
        Propagates the state forward in time using a process model. The user
        must provide the current state, input, and time interval

        .. note::
            If the time interval ``dt`` is not provided in the arguments, it will
            be taken as the difference between the input stamp and the state stamp.

        Parameters
        ----------
        x : MixtureState
            The current states and their associated weights.
        u : Input
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, dt will be calculated
            with ``dt = u.stamp - x.state.stamp``.
        Returns
        -------
        MixtureState
            New predicted states with associated weights.
        """

        n_modes = len(x.model_states)

        x_check = []
        for i in range(n_modes):
            x_check.append(self.ekf.predict(x.model_states[i], u, dt))
        return MixtureState(x_check, x.model_probabilities)
    
    def correct(
        self,
        x: MixtureState,
        y: Measurement,
        u: Input,
    ) -> MixtureState:
        """
        Corrects the state estimate using a measurement. The user must provide
        the current state and measurement.

        Parameters
        ----------
        x : MixtureState
            The current states and their associated weights.
        y : Measurement
            Measurement to correct the state estimate.

        Returns
        -------
        MixtureState
            Corrected states with associated weights.
        """
        x_check = x.copy()
        n_modes = len(x.model_states)
        weights_check = x.model_probabilities.copy()

        x_hat = []
        weights_hat = np.zeros(n_modes)
        for i in range(n_modes):
            x, details_dict = self.ekf.correct(x_check.model_states[i], y, u, 
                                               output_details=True)
            x_hat.append(x)
            z = details_dict["z"]
            S = details_dict["S"]
            model_likelihood = multivariate_normal.pdf(
                z.ravel(), mean=np.zeros(z.shape), cov=S
            )
            weights_hat[i] = weights_check[i] * model_likelihood

        # If all model likelihoods are zero to machine tolerance, np.sum(mu_k)=0 and it fails
        # Add this fudge factor to get through those cases.
        if np.allclose(weights_hat, np.zeros(weights_hat.shape)):
            weights_hat = 1e-10 * np.ones(weights_hat.shape)

        weights_hat = weights_hat / np.sum(weights_hat)
            
        return MixtureState(x_hat, weights_hat)    


def run_gsf_filter(
    filter: GaussianSumFilter,
    x0: StateWithCovariance,
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
    x0 : StateWithCovariance
        _description_
    input_data : List[Input]
        _description_
    meas_data : List[Measurement]
        _description_
    """
    x = x0.copy()
    if x.stamp is None:
        raise ValueError("x0 must have a valid timestamp.")

    # Sort the data by time
    input_data.sort(key=lambda x: x.stamp)
    meas_data.sort(key=lambda x: x.stamp)

    # Remove all that are before the current time
    for idx, u in enumerate(input_data):
        if u.stamp >= x.stamp:
            input_data = input_data[idx:]
            break

    for idx, y in enumerate(meas_data):
        if y.stamp >= x.stamp:
            meas_data = meas_data[idx:]
            break

    meas_idx = 0
    if len(meas_data) > 0:
        y = meas_data[meas_idx]

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