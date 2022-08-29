from .types import (
    StampedValue,
    State,
    ProcessModel,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from scipy.stats.distributions import chi2


def check_outlier(error: np.ndarray, covariance: np.ndarray):
    """
    Performs the Normalized-Innovation-Squared (NIS) test to identify
    an outlier.
    """
    error = error.reshape((-1, 1))
    md = np.ndarray.item(error.T @ np.linalg.solve(covariance, error))
    if md > chi2.ppf(0.99, df=error.size):
        is_outlier = True
    else:
        is_outlier = False

    return is_outlier


class ExtendedKalmanFilter:
    """
    On-manifold nonlinear Kalman filter.
    """
    __slots__ = ["process_model", "_u"]

    def __init__(self, process_model: ProcessModel):
        self.process_model = process_model
        self._u = None

    def predict(
        self,
        x: StateWithCovariance,
        u: StampedValue,
        dt=None,
        x_jac: State = None,
        use_last_input=True,
    ):
        """
        Propagates the state forward in time using a process model.

        Optionally, provide `x_jac` as the evaluation point for Jacobian and
        covariance functions.
        """

        # Make a copy so we dont modify the input
        x = x.copy()

        # If state has no time stamp, load from measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = u.stamp

        if dt is None:
            dt = u.stamp - x.state.stamp

        # Load dedicated jacobian evaluation point if user specified.
        if x_jac is None:
            x_jac = x.state

        # Use current input provided, or the one previously.
        if use_last_input:
            u_eval = self._u
        else:
            u_eval = u

        if u_eval is not None:
            A = self.process_model.jacobian(x_jac, u_eval, dt)
            Q = self.process_model.covariance(x_jac, u_eval, dt)
            x.state = self.process_model.evaluate(x.state, u_eval, dt)
            x.covariance = A @ x.covariance @ A.T + Q
            x.symmetrize()
            x.state.stamp += dt

        self._u = u

        return x

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        x_jac: State = None,
        reject_outlier=False,
    ):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Optionally, provide `x_jac` as the evaluation point for Jacobian and
        covariance functions.
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt > 0 and self._u is not None:
                x = self.predict(x, self._u, dt)

        if x_jac is None:
            x_jac = x.state
        P = x.covariance
        R = np.atleast_2d(y.model.covariance(x_jac))
        G = np.atleast_2d(y.model.jacobian(x_jac))
        y_check = y.model.evaluate(x.state)
        z = y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))
        S = G @ P @ G.T + R

        outlier = False

        # Test for outlier if requested.
        if reject_outlier:
            outlier = check_outlier(z, S)

        if not outlier:

            # Do the correction
            K = np.linalg.solve(S.T, (P @ G.T).T).T
            dx = K @ z
            x.state.plus(dx)
            x.covariance = (np.identity(x.state.dof) - K @ G) @ P
            x.symmetrize()

        return x


class IteratedKalmanFilter(ExtendedKalmanFilter):
    """
    On-manifold iterated extended Kalman filter.
    """
    def __init__(
        self,
        process_model: ProcessModel,
        step_tol=1e-4,
        max_iters=100,  # TODO. implement max iters
        line_search=True,  # TODO implement line search
    ):
        super(IteratedKalmanFilter, self).__init__(process_model)
        self.step_tol = step_tol
        self.max_iters = max_iters

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        x_jac: State = None,
        reject_outlier=False,
    ):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Optionally, provide `x_jac` as the evaluation point for Jacobian and
        covariance functions.
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        # If state has no time stamp, load form measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt > 0 and self._u is not None:
                x = self.predict(x, self._u, dt)

        dx = 10
        x_op = x.state.copy()  # Operating point
        while np.linalg.norm(dx) > self.step_tol:

            # Load a dedicated state evaluation point for jacobian
            # if the user supplied it.
            if x_jac is not None:
                x_op_jac = x_jac
            else:
                x_op_jac = x_op

            R = np.atleast_2d(y.model.covariance(x_op_jac))
            G = np.atleast_2d(y.model.jacobian(x_op_jac))
            y_check = y.model.evaluate(x_op)
            z = y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))
            S = G @ x.covariance @ G.T + R
            S = 0.5 * (S + S.T)
            e = x.state.minus(x_op).reshape((-1, 1))
            

            # Test for outlier if requested.
            outlier = False
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:
                K = np.linalg.solve(S.T, (x.covariance @ G.T).T).T
                dx = e + K @ (z - G @ e)
                x_op.plus(0.2 * dx)
            else:
                break

        x.state = x_op
        x.covariance = (np.identity(x.state.dof) - K @ G) @ x.covariance
        x.symmetrize()
        return x

    def _cost(
        self,
        prior_error: np.ndarray,
        prior_covariance: np.ndarray,
        meas_error: np.ndarray,
        meas_covariance: np.ndarray,
    ):
        e = prior_error
        P = prior_covariance
        z = meas_error
        R = meas_covariance
        cost_prior = np.ndarray.item(0.5 * e.T @ np.linalg.solve(P, e))
        cost_meas = np.ndarray.item(0.5 * z.T @ np.linalg.solve(R, z))
        return cost_prior + cost_meas, cost_prior, cost_meas
