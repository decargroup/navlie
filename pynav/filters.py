from .types import StampedValue, State, ProcessModel, Measurement
import numpy as np
from scipy.stats.distributions import chi2


def check_outlier(error: np.ndarray, covariance: np.ndarray):
    error = error.reshape((-1, 1))
    md = np.asscalar(error.T @ np.linalg.solve(covariance, error))
    if md > chi2.ppf(0.99, df=error.size):
        is_outlier = True
    else:
        is_outlier = False

    return is_outlier


class ExtendedKalmanFilter:
    """
    On-manifold nonlinear Kalman filter.
    """

    def __init__(self, x0: State, P0: np.ndarray, process_model: ProcessModel):
        self.process_model = process_model
        self.reset(x0, P0)

    def reset(self, x0: State, P0: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self._u = None

    def predict(
        self, u: StampedValue, dt=None, x_jac: State = None, use_last_input=True
    ):
        """
        Propagates the state forward in time using a process model.

        Optionally, provide `x_jac` as the evaluation point for Jacobian and
        covariance functions.
        """
        if self.x.stamp is None:
            self.x.stamp = u.stamp

        if dt is None:
            dt = u.stamp - self.x.stamp

        if x_jac is None:
            x_jac = self.x

        if use_last_input:
            u_eval = self._u
        else:
            u_eval = u

        if u_eval is not None:
            A = self.process_model.jacobian(x_jac, u_eval, dt)
            Q = self.process_model.covariance(x_jac, u_eval, dt)
            self.x = self.process_model.evaluate(self.x, u_eval, dt)
            self.P = A @ self.P @ A.T + Q
            self.P = 0.5 * (self.P + self.P.T)
            self.x.stamp += dt

        self._u = u

    def correct(
        self, y: Measurement, x_jac: State = None, reject_outlier=False
    ):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Optionally, provide `x_jac` as the evaluation point for Jacobian and
        covariance functions.
        """
        if self.x.stamp is None:
            self.x.stamp = y.stamp

        if y.stamp is not None:
            dt = y.stamp - self.x.stamp
            if dt > 0 and self._u is not None:
                self.predict(self._u, dt)

        if x_jac is None:
            x_jac = self.x

        R = np.atleast_2d(y.model.covariance(x_jac))
        G = np.atleast_2d(y.model.jacobian(x_jac))
        y_hat = y.model.evaluate(self.x)
        z = y.value.reshape((-1, 1)) - y_hat.reshape((-1, 1))
        S = G @ self.P @ G.T + R

        outlier = False

        # Test for outlier if requested.
        if reject_outlier:
            outlier = check_outlier(z, S)

        if not outlier:
            K = np.linalg.solve(S.T, (self.P @ G.T).T).T
            # K = (self.P @ G.T) @ np.linalg.inv(S)

            self.P = (np.identity(self.P.shape[0]) - K @ G) @ self.P
            self.P = 0.5 * (self.P + self.P.T)
            dx = K @ z
            self.x.plus(dx)


class IteratedKalmanFilter(ExtendedKalmanFilter):
    def __init__(
        self,
        x0: State,
        P0: np.ndarray,
        process_model: ProcessModel,
        step_tol=1e-4,
    ):
        super(IteratedKalmanFilter, self).__init__(x0, P0, process_model)
        self.step_tol = step_tol

    def correct(
        self, y: Measurement, x_jac: State = None, reject_outlier=False
    ):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Optionally, provide `x_jac` as the evaluation point for Jacobian and
        covariance functions.
        """

        # If state has no time stamp, load form measurement.
        # usually only happens on estimator start-up
        if self.x.stamp is None:
            self.x.stamp = y.stamp

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - self.x.stamp
            if dt > 0 and self._u is not None:
                self.predict(self._u, dt)

        dx = 10
        x_op = self.x.copy()  # Operating point
        cost_old = 0
        while np.linalg.norm(dx) > self.step_tol:

            # Load a dedicated state evaluation point for jacobian 
            # if the user supplied it.
            if x_jac is not None:
                x_op_jac = x_jac.copy()
            else:
                x_op_jac = x_op

            R = np.atleast_2d(y.model.covariance(x_op_jac))
            G = np.atleast_2d(y.model.jacobian(x_op_jac))
            y_hat = y.model.evaluate(x_op)
            z = y.value.reshape((-1, 1)) - y_hat.reshape((-1, 1))
            S = G @ self.P @ G.T + R
            S = 0.5 * (S + S.T)
            e = self.x.minus(x_op).reshape((-1, 1))
            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:
                K = np.linalg.solve(S.T, (self.P @ G.T).T).T
                dx = e + K @ (z - G @ e)
                x_op.plus(0.1 * dx)
            else:
                break


            
            cost_prior = np.asscalar(0.5 * e.T @ np.linalg.solve(self.P, e))
            cost_meas = np.asscalar(0.5 * z.T @ np.linalg.solve(R, z))
            cost = cost_prior + cost_meas
            print(
                "Prior err.: {0:.4e}  Meas err.: {1:.4e}  Total: {2:.4e} Change: {3:.4e}".format(
                    cost_prior, cost_meas, cost, cost - cost_old
                )
            )
            cost_old = cost


        self.x = x_op
        self.P = (np.identity(self.P.shape[0]) - K @ G) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        print(y.stamp)
