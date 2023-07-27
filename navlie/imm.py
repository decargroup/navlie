from typing import List

from navlie.types import (
    Input,
    State,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from scipy.stats import multivariate_normal
from navlie.utils import GaussianResultList, GaussianResult
from navlie.filters import ExtendedKalmanFilter


def gaussian_mixing_vectorspace(
    weights: List[float], means: List[np.ndarray], covariances: List[np.ndarray]
):
    """Calculate the mean and covariance of a Gaussian mixture on a vectorspace.

    Parameters
    ----------
    weights : List[Float]
        Weights corresponding to each Gaussian.
    means: List[np.ndarray]
        List containing the means of each Gaussian mixture
    covariances
        List containing covariances of each Gaussian mixture

    Returns
    -------
    np.ndarray
        Mean of Gaussian mixture
    np.ndarray
        Covariance of Gaussian mixture
    """

    x_bar = np.zeros(means[0].shape)
    P_bar = np.zeros(covariances[0].shape)

    for (weight, x) in zip(weights, means):
        x_bar = x_bar + weight * x

    for (weight, x, P) in zip(weights, means, covariances):
        dx = (x - x_bar).reshape(-1, 1)
        P_bar = P_bar + weight * P + weight * dx @ dx.T

    return x_bar, P_bar


def reparametrize_gaussians_about_X_par(
    X_par: State, X_list: List[StateWithCovariance]
):
    """Reparametrize each Lie group Gaussian in X_list about X_par.
    A Lie group Gaussian is only relevant in the tangent space of its own mean.
    To mix Lie group Gaussians, a specific X, X_par, must be chosen to expand
    a tangent space around. Once expanded in this common tangent space,
    the Gaussians may be mixed in a vector space fashion.

    Parameters
    ----------
    X_par : State
        Each member of X_list will be reparametrized as a Gaussian
            in the tangent space of X_par.
    X_list : List[StateWithCovariance]
        List of Lie group Gaussians to be reparametrized.

    Returns
    -------
    List[np.ndarray]
        Tangent space of X_par mean of each element of X_list.
    List[np.ndarray]
        Tangent space of X_par covariance of each element of X_list
    """
    means_reparametrized = []
    covariances_reparametrized = []

    for X in X_list:
        mu = X.state.minus(X_par)
        Jinv = X.state.minus_jacobian(X_par)
        Sigma = Jinv @ X.covariance @ Jinv.T
        means_reparametrized.append(mu)
        covariances_reparametrized.append(Sigma)

    return means_reparametrized, covariances_reparametrized


def update_X(X: State, mu: np.ndarray, P: np.ndarray):
    """Given a Lie group Gaussian with mean mu and covariance P, expressed in the tangent space of X,
    compute Lie group StateAndCovariance X_hat such that the Lie algebra Gaussian
    around X_hat has zero mean.

    Parameters
    ----------
    X : State
        A Lie group element.
    mu : np.ndarray
        Mean of Gaussian in tangent space of X
    P: np.ndarray
        Covariance of Gaussian in tangent space of X

    Returns
    -------
    StateWithCovariance
        StateWithCovariance whose state is a Lie group element.
    """
    X_hat = StateWithCovariance(X, np.zeros((X.dof, X.dof)))
    X_hat.state = X_hat.state.plus(mu)

    J = X.plus_jacobian(mu)

    X_hat.covariance = J @ P @ J.T

    return X_hat


def gaussian_mixing(weights: List[float], x_list: List[StateWithCovariance]):
    """A Gaussian mixing method that handles both vectorspace Gaussians
        and Gaussians on Lie groups.

    Parameters
    ----------
    weights : List[Float]
        Weights of Gaussians to be mixed.
    x_list : List[StateWithCovariance]
        List of Gaussians to be mixed.
    Returns
    -------
    StateWithCovariance
        The mixed Gaussian
    """
    max_idx = np.argmax(np.array(weights))
    X_par = x_list[max_idx].state
    mu_repar, P_repar = reparametrize_gaussians_about_X_par(X_par, x_list)
    x_bar, P_bar = gaussian_mixing_vectorspace(weights, mu_repar, P_repar)
    X_mix = update_X(X_par, x_bar, P_bar)
    return X_mix


class IMMState:
    __slots__ = ["model_states", "model_probabilities"]

    def __init__(
        self,
        model_states: List[StateWithCovariance],
        model_probabilities: List[float],
    ):
        self.model_states = model_states
        self.model_probabilities = model_probabilities

    def copy(self) -> "IMMState":
        return IMMState(
            self.model_states.copy(), self.model_probabilities.copy()
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
        self.model_probabilities = []
        # Turn list of "probability at time step" into
        # list of "probability of model"
        n_models = result_list[0].model_probabilities.shape[0]
        for lv1 in range(n_models):
            self.model_probabilities.append(
                np.array([r.model_probabilities[lv1] for r in result_list])
            )


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

    """

    def __init__(self, kf_list: List[ExtendedKalmanFilter], Pi: np.ndarray):
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
        x: IMMState,
    ):
        """The interaction (mixing) step of the IMM.

        Parameters
        ----------
        x : IMMState

        Returns
        -------
        IMMState
        """

        x_km_models = x.model_states.copy()
        mu_models = np.array(x.model_probabilities)

        n_modes = self.Pi.shape[0]
        c = self.Pi.T @ mu_models.reshape(-1, 1)

        mu_mix = np.zeros((n_modes, n_modes))
        for i in range(n_modes):
            for j in range(n_modes):
                mu_mix[i, j] = 1.0 / c[j] * self.Pi[i, j] * mu_models[i]
        x_mix = []

        for j in range(n_modes):
            weights = list(mu_mix[:, j])
            x_mix.append(gaussian_mixing(weights, x_km_models))

        return IMMState(x_mix, mu_models)

    def predict(self, x_km: IMMState, u: Input, dt: float):
        """Carries out prediction step for each model of the IMM.

        Parameters
        ----------
        x_km : IMMState
            Model estimates from previous timestep, after mixing.
        u : Input
            Input
        dt : Float
            Timestep

        Returns
        -------
        IMMState
        """
        x_km_models = x_km.model_states.copy()
        x_check = []
        for lv1, kf in enumerate(self.kf_list):
            x_check.append(kf.predict(x_km_models[lv1], u, dt))
        return IMMState(x_check, x_km.model_probabilities)

    def correct(
        self,
        x_check: IMMState,
        y: Measurement,
        u: Input,
    ):
        """Carry out the correction step for each model and update model probabilities.

        Parameters
        ----------
        x_check: IMMState
        mu_km_models : List[Float]
            Probabilities for each model from previous timestep.
        y : Measurement
            Measurement to be fused into the current state estimate.
        u: Input
            Most recent input, to be used to predict the state forward
            if the measurement stamp is larger than the state stamp.


        Returns
        -------
        IMMState
            Corrected state estimates and probabilities
        """
        x_models_check = x_check.model_states.copy()
        mu_km_models = x_check.model_probabilities.copy()
        n_modes = len(x_models_check)
        mu_k = np.zeros(n_modes)

        # Compute each model's normalization constant
        c_bar = np.zeros(n_modes)
        for i in range(n_modes):
            for j in range(n_modes):
                c_bar[j] = c_bar[j] + self.Pi[i, j] * mu_km_models[i]

        # Correct and update model probabilities
        x_hat = []
        for lv1, kf in enumerate(self.kf_list):
            x, details_dict = kf.correct(
                x_models_check[lv1], y, u, output_details=True
            )
            x_hat.append(x)
            z = details_dict["z"]
            S = details_dict["S"]
            z = z.ravel()
            model_likelihood = multivariate_normal.pdf(
                z, mean=np.zeros(z.shape), cov=S
            )
            mu_k[lv1] = model_likelihood * c_bar[lv1]

        # If all model likelihoods are zero to machine tolerance, np.sum(mu_k)=0 and it fails
        # Add this fudge factor to get through those cases.
        if np.allclose(mu_k, np.zeros(mu_k.shape)):
            mu_k = 1e-10 * np.ones(mu_k.shape)

        mu_k = mu_k / np.sum(mu_k)

        return IMMState(x_hat, mu_k)


def run_interacting_multiple_model_filter(
    filter: InteractingModelFilter,
    x0: State,
    P0: np.ndarray,
    input_data: List[Input],
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

    results_list = []
    n_models = filter.Pi.shape[0]

    x = IMMState(
        [StateWithCovariance(x0, P0)] * n_models,
        1.0 / n_models * np.array(np.ones(n_models)),
    )
    for k in range(len(input_data) - 1):
        results_list.append(x)
        u = input_data[k]
        # Fuse any measurements that have occurred.
        if len(meas_data) > 0:
            while y.stamp < input_data[k + 1].stamp and meas_idx < len(
                meas_data
            ):

                x = filter.interaction(x)
                x = filter.correct(x, y, u)
                meas_idx += 1
                if meas_idx < len(meas_data):
                    y = meas_data[meas_idx]
        dt = input_data[k + 1].stamp - x.model_states[0].stamp
        x = filter.predict(x, u, dt)

    return results_list
