"""
Collection of function required for mixing Gaussian distributions.
"""

from typing import List

from navlie.types import (
    State,
    Input,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from navlie.lib import MixtureState


# TODO. The IMM seems to have an issue when the user accidently modifies the
# provided state in the process model.



def gaussian_mixing_vectorspace(
    weights: List[float], means: List[np.ndarray], covariances: List[np.ndarray]
):
    """
    Calculate the mean and covariance of a Gaussian mixture on a vectorspace.

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

    for weight, x in zip(weights, means):
        x_bar = x_bar + weight * x

    for weight, x, P in zip(weights, means, covariances):
        dx = (x - x_bar).reshape(-1, 1)
        P_bar = P_bar + weight * P + weight * dx @ dx.T

    return x_bar, P_bar


def reparametrize_gaussians_about_X_par(
    X_par: State, X_list: List[StateWithCovariance]
):
    """
    Reparametrize each Lie group Gaussian in X_list about X_par.
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
    """
    Given a Lie group Gaussian with mean mu and covariance P, expressed in the
    tangent space of X, compute Lie group StateAndCovariance X_hat such that the
    Lie algebra Gaussian around X_hat has zero mean.

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
