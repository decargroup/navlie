from typing import List, Tuple
from navlie.types import (
    Input,
    State,
    ProcessModel,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from scipy.stats.distributions import chi2
from numpy.polynomial.hermite_e import hermeroots
from math import factorial
from scipy.special import eval_hermitenorm
import scipy.linalg as la
from tqdm import tqdm


def check_outlier(error: np.ndarray, covariance: np.ndarray):
    """
    Performs the Normalized-Innovation-Squared (NIS) test to identify
    an outlier.
    """
    error = error.reshape((-1, 1))
    nis = np.ndarray.item(error.T @ np.linalg.solve(covariance, error))
    if nis > chi2.ppf(0.99, df=error.size):
        is_outlier = True
    else:
        is_outlier = False

    return is_outlier


def mean_state(x_array: List[State], weights: np.ndarray) -> State:
    """Computes a weighted mean of a list of State instances
    in an iterated manner, until reaching a maximun number of
    iterations or a small update.

    Parameters
    ----------
    x_array : List[State]
        List of states to be averaged. They should be of the same type
    weights : np.ndarray
        weights associated to each state

    Returns
    -------
    State
        Returns the mean state.
    """
    x_0 = x_array[0]
    n = len(x_array)

    x_mean = x_0.copy()

    iter = 0
    err = 1
    while np.linalg.norm(err) > 1e-6 and iter <= 50:
        err = np.zeros(x_0.dof)
        for i in range(n):
            err += weights[i] * (x_array[i].minus(x_mean)).ravel()

        x_mean = x_mean.plus(err)
        iter += 1

    return x_mean


class ExtendedKalmanFilter:
    """
    On-manifold nonlinear Kalman filter.
    """

    __slots__ = ["process_model", "reject_outliers"]

    def __init__(self, process_model: ProcessModel, reject_outliers=False):
        """
        Parameters
        ----------
        process_model : ProcessModel
            process model to be used in the prediction step
        reject_outliers : bool, optional
            whether to apply the NIS test to measurements, by default False
        """
        self.process_model = process_model
        self.reject_outliers = reject_outliers

    def predict(
        self,
        x: StateWithCovariance,
        u: Input,
        dt: float = None,
        x_jac: State = None,
        output_details: bool = False,
    ) -> StateWithCovariance:
        """
        Propagates the state forward in time using a process model. The user
        must provide the current state, input, and time interval

        .. note::
            If the time interval `dt` is not provided in the arguments, it will
            be taken as the difference between the input stamp and the state stamp.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        u : Input
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, dt will be calculated
            with `dt = u.stamp - x.state.stamp`.
        x_jac : State, optional
            Evaluation point for the process model Jacobian. If not provided, the
            current state estimate will be used.

        Returns
        -------
        StateWithCovariance
            New predicted state
        """

        # Make a copy so we dont modify the input
        x_new = x.copy()

        # If state has no time stamp, load from measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            t_km1 = u.stamp
        else:
            t_km1 = x.state.stamp

        if dt is None:
            dt = u.stamp - t_km1

        if dt < 0:
            raise RuntimeError("dt is negative!")

        # Load dedicated jacobian evaluation point if user specified.
        if x_jac is None:
            x_jac = x.state

        details_dict = {}
        if u is not None:
            A = self.process_model.jacobian(x_jac, u, dt)
            Q = self.process_model.covariance(x_jac, u, dt)
            x_new.state = self.process_model.evaluate(x.state, u, dt)
            x_new.covariance = A @ x.covariance @ A.T + Q
            x_new.symmetrize()
            x_new.state.stamp = t_km1 + dt

            details_dict = {"A": A, "Q": Q}

        if output_details:
            return x_new, details_dict
        else:
            return x_new

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        u: Input,
        x_jac: State = None,
        reject_outlier: bool = None,
        output_details: bool = False,
    ) -> StateWithCovariance:
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.
        If a measurement model returns `None` from its `evaluate()` method,
        the measurement will not be fused.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        y : Measurement
            Measurement to be fused into the current state estimate.
        u: Input
            Most recent input, to be used to predict the state forward
            if the measurement stamp is larger than the state stamp. If set to
            None, no prediction will be performed and the correction will
            just be done with the current state estimate.
        x_jac : State, optional
            valuation point for the process model Jacobian. If not provided, the
            current state estimate will be used.
        reject_outlier : bool, optional
            Whether to apply the NIS test to this measurement, by default None,
            in which case the value of `self.reject_outliers` will be used.
        output_details : bool, optional
            Whether to output intermediate computation results (innovation,
            innovation covariance) in an additional returned dict.
        Returns
        -------
        StateWithCovariance
            The corrected state estimate
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # Load default outlier rejection option
        if reject_outlier is None:
            reject_outlier = self.reject_outliers

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt < -1e10:
                raise RuntimeError(
                    "Measurement stamp is earlier than state stamp"
                )
            elif u is not None and dt > 1e-11:
                x = self.predict(x, u, dt)

        if x_jac is None:
            x_jac = x.state
        y_check = y.model.evaluate(x.state)

        details_dict = {}
        if y_check is not None:
            P = x.covariance
            R = np.atleast_2d(y.model.covariance(x_jac))
            G = np.atleast_2d(y.model.jacobian(x_jac))
            z = y.minus(y_check)
            S = G @ P @ G.T + R

            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:
                # Do the correction
                K = np.linalg.solve(S.T, (P @ G.T).T).T
                dx = K @ z
                x.state = x.state.plus(dx)
                x.covariance = (np.identity(x.state.dof) - K @ G) @ P
                x.symmetrize()

            details_dict = {"z": z, "S": S, "is_outlier": outlier}

        if output_details:
            return x, details_dict
        else:
            return x


class IteratedKalmanFilter(ExtendedKalmanFilter):
    """
    On-manifold iterated extended Kalman filter.
    """

    __slots__ = [
        "process_model",
        "reject_outliers",
        "step_tol",
        "max_iters",
        "line_search",
    ]

    def __init__(
        self,
        process_model: ProcessModel,
        step_tol=1e-4,
        max_iters=200,
        line_search=True,
        reject_outliers=False,
    ):
        super(IteratedKalmanFilter, self).__init__(process_model)
        self.step_tol = step_tol
        self.max_iters = max_iters
        self.reject_outliers = reject_outliers
        self.line_search = line_search

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        u: Input,
        x_jac: State = None,
        reject_outlier=None,
    ):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        u: Input
            Most recent input, to be used to predict the state forward
            if the measurement stamp is larger than the state stamp.
        y : Measurement
            Measurement to be fused into the current state estimate.
        x_jac : State, optional
            valuation point for the process model Jacobian. If not provided, the
            current state estimate will be used.
        reject_outlier : bool, optional
            Whether to apply the NIS test to this measurement, by default None,
            in which case the value of `self.reject_outliers` will be used.

        Returns
        -------
        StateWithCovariance
            The corrected state estimate
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        # Load default outlier rejection option
        if reject_outlier is None:
            reject_outlier = self.reject_outliers

        # If state has no time stamp, load form measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt < 0:
                raise RuntimeError(
                    "Measurement stamp is earlier than state stamp"
                )
            elif dt > 0 and u is not None:
                x = self.predict(x, u, dt)

        dx = np.zeros((x.state.dof,))
        x_op = x.state.copy()  # Operating point
        count = 0
        while count < self.max_iters:
            # Load a dedicated state evaluation point for jacobian
            # if the user supplied it.

            if x_jac is not None:
                x_op_jac = x_jac
            else:
                x_op_jac = x_op

            R = np.atleast_2d(y.model.covariance(x_op))
            G = np.atleast_2d(y.model.jacobian(x_op_jac))
            y_check = y.model.evaluate(x_op)
            z = y.minus(y_check)
            e = x_op.minus(x.state).reshape((-1, 1))
            J = x_op.plus_jacobian(e)

            P_inv = np.linalg.inv(x.covariance)
            R_inv = np.linalg.inv(R)
            cost_old = np.ndarray.item(0.5 * (e.T @ P_inv @ e + z.T @ R_inv @ z))
            P = J @ x.covariance @ J.T

            # Compute covariance of innovation
            S = G @ P @ G.T + R
            S = 0.5 * (S + S.T)

            # Test for outlier if requested.
            outlier = False
            if reject_outlier:
                outlier = check_outlier(z, S)

            # If outlier, immediately exit.
            if outlier:
                break

            K = np.linalg.solve(S.T, (P @ G.T).T).T
            dx = -J @ e + K @ (z + G @ J @ e)

            # If step direction is small already, exit loop.
            if np.linalg.norm(dx) < self.step_tol:
                break

            # Perform backtracking line search
            if self.line_search:
                alpha = 1
                step_accepted = False
                while not step_accepted and alpha > self.step_tol:
                    x_new = x_op.plus(alpha * dx)
                    y_check = y.model.evaluate(x_new)
                    z_new = y.minus(y_check)
                    e_new = x_new.minus(x.state).reshape((-1, 1))
                    cost_new = np.ndarray.item(
                        0.5
                        * (e_new.T @ P_inv @ e_new + z_new.T @ R_inv @ z_new)
                    )
                    if cost_new < cost_old:
                        step_accepted = True

                    else:
                        alpha *= 0.9
            else:
                # If line search is disabled, step is accepted by default.
                # Take full step.
                step_accepted = True
                x_new = x_op.plus(dx)

            # If step accepted, set new operating point. Otherwise,
            # immediately halt iterations, x_op will be the latest valid step.
            if step_accepted:
                x_op = x_new
            else:
                break

            count += 1

        if not outlier:
            # We need to recompute some stuff for covariance
            # calculation purposes.
            if x_jac is not None:
                x_op_jac = x_jac
            else:
                x_op_jac = x_op

            # Re-evaluate the jacobians at our latest operating point
            G = np.atleast_2d(y.model.jacobian(x_op_jac))
            R = np.atleast_2d(y.model.covariance(x_op_jac))
            e = x_op.minus(x.state).reshape((-1, 1))
            J = x_op.plus_jacobian(e)
            P = J @ x.covariance @ J.T
            S = G @ P @ G.T + R
            S = 0.5 * (S + S.T)
            K = np.linalg.solve(S.T, (P @ G.T).T).T
            x.state = x_op
            x.covariance = (np.identity(x.state.dof) - K @ G) @ (
                J @ x.covariance @ J.T
            )

        x.symmetrize()

        return x

def generate_sigmapoints(
    dof: int, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates unit sigma points from three available
    methods.

    Parameters
    ----------
    dof : int
        dof of the state involved
    method : str
        Method for generating sigma points
        'unscented': Unscented method
        'cubature': cubature method
        'gh': Gauss-Hermite method


    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        returns the unit sigma points and the weights
    """
    if method == "unscented":
        kappa = 2
        sigma_points = np.sqrt(dof + kappa) * np.block(
            [[np.zeros((dof, 1)), np.eye(dof), -np.eye(dof)]]
        )

        w = 1 / (2 * (dof + kappa)) * np.ones((2 * dof + 1))
        w[0] = kappa / (dof + kappa)

    elif method == "cubature":
        sigma_points = np.sqrt(dof) * np.block([[np.eye(dof), -np.eye(dof)]])

        w = 1 / (2 * dof) * np.ones((2 * dof))

    elif method == "gh":
        p = 3

        c = np.zeros(p + 1)
        c[-1] = 1
        sigma_points_scalar = hermeroots(c)
        weights_scalar = np.zeros(p)
        for i in range(p):
            weights_scalar[i] = factorial(p) / (
                (p * eval_hermitenorm(p - 1, sigma_points_scalar[i])) ** 2
            )

        # Generate all p^dof collections of indexes by
        # transforming numbers 0...p^dof-1) into p-base system
        # and by adding 1 to each digit
        num = np.linspace(0, p ** (dof) - 1, p ** (dof))
        ind = np.zeros((dof, p**dof))
        for i in range(dof):
            ind[i, :] = num % p
            num = num // p

        sigma_points = np.zeros((dof, p**dof))
        w = np.zeros(p**dof)
        for i in range(p**dof):
            w[i] = 1
            sigma_point = []
            for j in range(dof):
                w[i] *= weights_scalar[int(ind[j, i])]
                sigma_point.append(sigma_points_scalar[int(ind[j, i])])
            sigma_points[:, i] = np.vstack(sigma_point).ravel()

    return sigma_points, w


class SigmaPointKalmanFilter:
    """
    On-manifold nonlinear Sigma Point Kalman filter.
    """

    __slots__ = [
        "process_model",
        "method",
        "reject_outliers",
        "iterate_mean",
        "_sigmapoint_cache",
    ]

    def __init__(
        self,
        process_model: ProcessModel,
        method: str = "unscented",
        reject_outliers=False,
        iterate_mean=True,
    ):
        """
        Parameters
        ----------
        process_model : ProcessModel
            process model to be used in the prediction step
        method : str
            method to generate the sigma points. Options are
                'unscented': unscented sigma points
                'cubature': cubature sigma points
                'gh': Gauss-hermite sigma points
        reject_outliers : bool, optional
            whether to apply the NIS test to measurements, by default False
        iterate_mean : bool, optional
            whether to compute the mean state with sigma points or
            by propagating \check {x_{k-1}} on the process model
        """
        self.process_model = process_model
        self.method = method
        self.reject_outliers = reject_outliers
        self.iterate_mean = iterate_mean
        self._sigmapoint_cache = {}

    def predict(
        self,
        x: StateWithCovariance,
        u: Input,
        dt: float = None,
        input_covariance: np.ndarray = None,
    ) -> StateWithCovariance:
        """
        Propagates the state forward in time using a process model. The user
        must provide the current state, input, and time interval

        .. note::
            If the time interval `dt` is not provided in the arguments, it will
            be taken as the difference between the input stamp and the state stamp.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        u : Input
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, dt will be calculated
            with `dt = u.stamp - x.state.stamp`.
        input_covariance: np.ndarray, optional
            Covariance associated to the inpu measurement. If not provided,
            it will be grabbed from u.covariance
        Returns
        -------
        StateWithCovariance
            New predicted state
        """

        # Make a copy so we dont modify the input
        x = x.copy()

        # If state has no time stamp, load from measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = u.stamp

        if input_covariance is None:
            if u.covariance is not None:
                input_covariance = u.covariance
            else:
                raise ValueError(
                    "Input covariance information must be provided."
                )

        if dt is None:
            dt = u.stamp - x.state.stamp

        if dt < 0:
            raise RuntimeError("dt is negative!")

        if u is not None:
            n_x = x.state.dof
            n_u = u.dof

            P = la.block_diag(x.covariance, input_covariance)

            P_sqrt = np.linalg.cholesky(P)

            n = n_x + n_u

            if (n, self.method) in self._sigmapoint_cache:
                unit_sigmapoints, w = self._sigmapoint_cache[(n, self.method)]
            else:
                unit_sigmapoints, w = generate_sigmapoints(n, self.method)
                self._sigmapoint_cache[(n, self.method)] = (unit_sigmapoints, w)

            sigmapoints = P_sqrt @ unit_sigmapoints

            n_sig = w.size
            # Propagate
            x_propagated = [
                self.process_model.evaluate(
                    x.state.plus(sp[0:n_x]),
                    u.plus(sp[n_x:]),
                    dt,
                )
                for sp in sigmapoints.T
            ]

            # Compute mean.
            if self.iterate_mean:
                x_mean = mean_state(x_propagated, w)
            else:
                x_mean = self.process_model.evaluate(x.state, u, dt)

            # Compute covariance
            P_new = np.zeros(x.covariance.shape)
            for i in range(n_sig):
                err = x_mean.minus(x_propagated[i])
                err = err.reshape((-1, 1))
                P_new += w[i] * err @ err.T

            x.state = x_mean
            x.covariance = P_new
            x.symmetrize()
            x.state.stamp += dt

        return x

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        u: Input,
        reject_outlier: bool = None,
    ) -> StateWithCovariance:
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        u: StampedValue
            Most recent input, to be used to predict the state forward
            if the measurement stamp is larger than the state stamp.
        y : Measurement
            Measurement to be fused into the current state estimate.
        reject_outlier : bool, optional
            Whether to apply the NIS test to this measurement, by default None,
            in which case the value of `self.reject_outliers` will be used.

        Returns
        -------
        StateWithCovariance
            The corrected state estimate
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # Load default outlier rejection option
        if reject_outlier is None:
            reject_outlier = self.reject_outliers

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt < 0:
                raise RuntimeError(
                    "Measurement stamp is earlier than state stamp"
                )
            elif u is not None:
                x = self.predict(x, u, dt)

        P_xx = x.covariance
        R = y.model.covariance(x.state)

        n_x = x.state.dof
        n_y = y.value.size

        if (n_x, self.method) in self._sigmapoint_cache:
            unit_sigmapoints, w = self._sigmapoint_cache[(n_x, self.method)]
        else:
            unit_sigmapoints, w = generate_sigmapoints(n_x, self.method)
            self._sigmapoint_cache[(n_x, self.method)] = (unit_sigmapoints, w)

        P_sqrt = np.linalg.cholesky(P_xx)
        sigmapoints = P_sqrt @ unit_sigmapoints

        n_sig = w.size

        y_check = y.model.evaluate(x.state)

        if y_check is not None:
            y_propagated = [
                y.model.evaluate(x.state.plus(sp)).ravel()
                for sp in sigmapoints.T
            ]

            # predicted measurement mean
            y_mean = np.zeros(n_y)
            for i in range(n_sig):
                y_mean += w[i] * y_propagated[i]

            # compute covariance of innovation and cross covariance
            Pyy = np.zeros((n_y, n_y))
            Pxy = np.zeros((n_x, n_y))
            for i in range(n_sig):
                err = y_propagated[i].reshape((-1, 1)) - y_mean.reshape((-1, 1))

                Pyy += w[i] * err @ err.T
                Pxy += w[i] * sigmapoints[:, i].reshape((-1, 1)) @ err.T

            Pyy += R

            z = y.minus(y_mean)

            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, Pyy)

            if not outlier:
                # Do the correction
                K = np.linalg.solve(Pyy.T, Pxy.T).T
                dx = K @ z
                x.state = x.state.plus(dx)
                x.covariance = P_xx - K @ Pxy.T
                x.symmetrize()

        return x


class UnscentedKalmanFilter(SigmaPointKalmanFilter):
    def __init__(
        self,
        process_model: ProcessModel,
        reject_outliers=False,
        iterate_mean=True,
    ):
        super().__init__(
            process_model=process_model,
            method="unscented",
            reject_outliers=reject_outliers,
            iterate_mean=iterate_mean,
        )

class CubatureKalmanFilter(SigmaPointKalmanFilter):
    def __init__(
        self,
        process_model: ProcessModel,
        reject_outliers=False,
        iterate_mean=True,
    ):
        super().__init__(
            process_model=process_model,
            method="cubature",
            reject_outliers=reject_outliers,
            iterate_mean=iterate_mean,
        )

class GaussHermiteKalmanFilter(SigmaPointKalmanFilter):
    def __init__(
        self,
        process_model: ProcessModel,
        reject_outliers=False,
        iterate_mean=True,
    ):
        super().__init__(
            process_model=process_model,
            method="gh",
            reject_outliers=reject_outliers,
            iterate_mean=iterate_mean,
        )


def run_filter(
    filter: ExtendedKalmanFilter,
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
    filter : ExtendedKalmanFilter
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
