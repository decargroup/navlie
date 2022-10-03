from logging import Filter
from typing import List

from .types import (
    StampedValue,
    State,
    ProcessModel,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from scipy.stats.distributions import chi2
from scipy.stats import multivariate_normal
from pynav.lib.states import VectorState
from pynav.filters import ExtendedKalmanFilter, check_outlier
from pynav.utils import GaussianResultList, GaussianResult


def gaussian_mixing_vectorspace(
    weights: List[float], x_list: List[StateWithCovariance]
):
    """Calculate the mean and covariance of a Gaussian mixture on a vectorspace.

    Parameters
    ----------
    weights : List[Float]
        Weights corresponding to each Gaussian.
    x_list : List[StateWithCovariance]
        Gaussians to be mixed, with each element of x_list containing the mean
        and covariance of the corresponding Gaussian. Each state must
        be a VectorState.

    Returns
    -------
    StateWithCovariance
        A single StateWithCovariance containing the mean and covariance of the mixed Gaussian.
    """
    x_bar = StateWithCovariance(
        VectorState(np.zeros(x_list[0].state.value.shape), stamp=x_list[0].state.stamp),
        np.zeros(x_list[0].covariance.shape),
    ).copy()

    for (weight, x) in zip(weights, x_list):
        x_bar.state.value = x_bar.state.value + weight * x.state.value

    for (weight, x) in zip(weights, x_list):
        dx = (x.state.value - x_bar.state.value).reshape(-1, 1)
        x_bar.covariance = x_bar.covariance + weight * x.covariance + weight * dx @ dx.T

    return x_bar


def reparametrize_gaussians_about_X_par(
    X_par: StateWithCovariance, X_list: List[StateWithCovariance]
):
    """Reparametrize each Lie group Gaussian in X_list about X_par.
    A Lie group Gaussian is only relevant in the tangent space of its own mean.
    To mix Lie group Gaussians, a specific X, X_par, must be chosen to expand
    a tangent space around. Once expanded in this common tangent space,
    the Gaussians may be mixed in a vector space fashion.

    Parameters
    ----------
    X_par : StateWithCovariance
        Each member of X_list will be reparametrized as a Gaussian
            in the tangent space of X_par.
    X_list : List[StateWithCovariance]
        List of Lie group Gaussians to be reparametrized.

    Returns
    -------
    List[StateWithCovariance]
        Where each state is a VectorState. These correspond to the mean and covariance
        of X_list, expressed in the tangent space of X_par.
    """
    x_reparametrized = []

    for X in X_list:
        mu = X_par.minus(X.state)
        J = X_par.jacobian(mu)
        Sigma = J @ X.covariance @ J.T
        mu = VectorState(mu, stamp=X_par.stamp)
        x_reparametrized.append(StateWithCovariance(mu, Sigma))

    return x_reparametrized


def update_X(X: State, x_hat: StateWithCovariance):
    """Given a Lie group Gaussian x_hat, expressed in the tangent space of X,
    compute Lie group StateAndCovariance X_hat such that the Lie algebra Gaussian
    around X_hat has zero mean.

    Parameters
    ----------
    X : State
        A Lie group element.
    x_hat : StateWithCovariance
        StateWithCovariance whose state is a VectorState.

    Returns
    -------
    StateWithCovariance
        StateWithCovariance whose state is a Lie group element.
    """
    X_hat = StateWithCovariance(X, np.zeros((X.dof, X.dof)))
    X_hat.state = X_hat.state.plus(x_hat.state.value)
    g = X.group

    if X.direction == "right":
        J = g.right_jacobian(x_hat.state.value)
        X_hat.covariance = J @ x_hat.covariance @ J.T

    return X_hat


def gaussian_mixing(weights: List[float], x_list: List[StateWithCovariance]):
    """A Gaussian mixing method that handles both vectorspace Gaussians
        and Gaussians on Lie groups.

    Parameters
    ----------
    weights : List[Float]
        Weights of Gaussians to be mixed.
    x_list : _List[StateWithCovariance]
        List of Gaussians to be mixed.
    Returns
    -------
    StateWithCovariance
        The mixed Gaussian
    """
    if isinstance(x_list[0].state, VectorState):
        X_mix = gaussian_mixing_vectorspace(weights, x_list)
    if not isinstance(x_list[0].state, VectorState):
        max_idx = np.argmax(np.array(weights))
        X_par = x_list[max_idx].state
        x_repar = reparametrize_gaussians_about_X_par(X_par, x_list)
        x_hat = gaussian_mixing_vectorspace(weights, x_repar)
        X_mix = update_X(X_par, x_hat)
    return X_mix


class ImmState:
    __slots__ = ["model_states", "model_probabilities"]

    def __init__(
        self, model_states: List[StateWithCovariance], model_probabilities: List[float]
    ):
        self.model_states = model_states
        self.model_probabilities = model_probabilities


class ImmResult(GaussianResult):
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

    def __init__(self, imm_estimate, state_true):
        super().__init__(
            gaussian_mixing(
                imm_estimate.model_probabilities, imm_estimate.model_states
            ),
            state_true,
        )
        self.model_probabilities = imm_estimate.model_probabilities


class ImmResultList(GaussianResultList):
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

    def __init__(self, result_list: List[ImmResult]):
        super().__init__(result_list)
        self.model_probabilities = []
        # Turn list of "probability at time step" into
        # list of "probability of model"
        N_MODELS = result_list[0].model_probabilities.shape[0]
        for lv1 in range(N_MODELS):
            self.model_probabilities.append(
                np.array([r.model_probabilities[lv1] for r in result_list])
            )


# TODO: Update docs
class InteractingModelFilter:
    """On-manifold Interacting Multiple Model Filter (IMM).
    References for the IMM:
     @article{blom1988interacting,
         author = {Blom, Henk A P and Bar-Shalom, Yaakov},
         journal = {IEEE transactions on Automatic Control},
         number = {8},
         pages = {780--783},
         publisher = {IEEE},
         title = {{The interacting multiple model algorithm for systems with Markovian switching coefficients}},
         volume = {33},
         year = {1988}
         }
     The IMM involves Gaussian mixtures.
     Reference for mixing Gaussians on manifolds:
     @article{7968489,
         author = {{\'{C}}esi{\'{c}}, Josip and Markovi{\'{c}}, Ivan and Petrovi{\'{c}}, Ivan},
         doi = {10.1109/LSP.2017.2723765},
         journal = {IEEE Signal Processing Letters},
         number = {11},
         pages = {1719--1723},
         title = {{Mixture Reduction on Matrix Lie Groups}},
         volume = {24},
         year = {2017}
         }
     TODOS: Implement methods for left perturbations.

    """

    def __init__(self, kf_list: List[Filter], Pi: np.ndarray):
        """Initialize InteractingModelFilter.

        Parameters
        ----------
        kf_list : List[ExtendedKalmanFilter]
            A list of filter instances which correspond to
            each model of the IMM.
        Pi : np.ndarray
            Probability transition matrix corresponding to the IMM models.
        """
        self.kf_list = kf_list
        self.Pi = Pi

    def interaction(
        self,
        x: ImmState,
    ):
        """The interaction (mixing) step of the IMM.

        Parameters
        ----------
        x : ImmState

        Returns
        -------
        ImmState
        """

        x_km_models = x.model_states.copy()
        mu_models = x.model_probabilities.copy()

        N_MODES = self.Pi.shape[0]
        c = self.Pi.T @ mu_models.reshape(-1, 1)

        mu_mix = np.zeros((N_MODES, N_MODES))
        for i in range(N_MODES):
            for j in range(N_MODES):
                mu_mix[i, j] = 1.0 / c[j] * self.Pi[i, j] * mu_models[i]
        x_mix = []

        for j in range(N_MODES):
            weights = list(mu_mix[:, j])
            x_mix.append(gaussian_mixing(weights, x_km_models))

        return ImmState(x_mix, mu_models)

    def predict(self, x_km: ImmState, u: StampedValue, dt: float):
        """Carries out prediction step for each model of the IMM.

        Parameters
        ----------
        x_km : ImmState
            Model estimates from previous timestep, after mixing.
        u : StampedValue
            Input
        dt : Float
            Timestep

        Returns
        -------
        ImmState
        """
        x_km_models = x_km.model_states.copy()
        x_check = []
        for lv1, kf in enumerate(self.kf_list):
            x_check.append(kf.predict(x_km_models[lv1], u, dt))
        return ImmState(x_check, x_km.model_probabilities)

    def correct(
        self,
        x_check: ImmState,
        y: Measurement,
        u: StampedValue,
    ):
        """Carry out the correction step for each model and update model probabilities.

        Parameters
        ----------
        x_check: ImmState
        mu_km_models : List[Float]
            Probabilities for each model from previous timestep.
        y : Measurement
            Measurement to be fused into the current state estimate.
        u: StampedValue
            Most recent input, to be used to predict the state forward
            if the measurement stamp is larger than the state stamp.


        Returns
        -------
        ImmState
            Corrected state estimates and probabilities
        """
        x_models_check = x_check.model_states.copy()
        mu_km_models = x_check.model_probabilities.copy()
        N_MODES = len(x_models_check)
        mu_k = np.zeros(N_MODES)

        # Compute each model's normalization constant
        c_bar = np.zeros(N_MODES)
        for i in range(N_MODES):
            for j in range(N_MODES):
                c_bar[j] = c_bar[j] + self.Pi[i, j] * mu_km_models[i]

        # Correct and update model probabilities
        x_hat = []
        for lv1, kf in enumerate(self.kf_list):
            x, details_dict = kf.correct(x_models_check[lv1], y, u, output_details=True)
            x_hat.append(x)
            z = details_dict["z"]
            S = details_dict["S"]
            model_likelihood = multivariate_normal.pdf(z, mean=np.zeros(z.shape), cov=S)
            mu_k[lv1] = model_likelihood * c_bar[lv1]

        # If all model likelihoods are zero to machine tolerance, np.sum(mu_k)=0 and it fails
        # Add this fudge factor to get through those cases.
        if np.allclose(mu_k, np.zeros(mu_k.shape)):
            mu_k = 1e-10 * np.ones(mu_k.shape)

        mu_k = mu_k / np.sum(mu_k)

        return ImmState(x_hat, mu_k)


def run_interacting_multiple_model_filter(
    filter,
    x0: State,
    P0: np.ndarray,
    input_data: List[StampedValue],
    meas_data: List[Measurement],
) -> List[StateWithCovariance]:
    """
    Executes an InteractingMultipleModel filter

    Parameters
    ----------
    filter: An InteractingModelFilter instance:
        _description_
    ProcessModel: Callable, must return a process model:
        _description_
    Q_profile: Callable, must return a square np.array compatible with process model:
        _description_
    x0 : State
        _description_
    P0 : np.ndarray
        _description_
    input_data : List[StampedValue]
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

    results_list = []
    N_MODELS = filter.Pi.shape[0]

    x = ImmState(
        [StateWithCovariance(x0, P0)] * N_MODELS,
        1.0 / N_MODELS * np.array(np.ones(N_MODELS)),
    )
    for k in range(len(input_data) - 1):
        results_list.append(x)
        u = input_data[k]
        # Fuse any measurements that have occurred.
        if len(meas_data) > 0:
            while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

                x = filter.interaction(x)
                x = filter.correct(x, y, u)
                meas_idx += 1
                if meas_idx < len(meas_data):
                    y = meas_data[meas_idx]
        dt = input_data[k + 1].stamp - x.model_states[0].stamp
        x = filter.predict(x, u, dt)

    return results_list
