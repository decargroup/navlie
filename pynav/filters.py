from typing import List

from pynav.lib.states import MatrixLieGroupState
from .types import (
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
        x = x.copy()

        # If state has no time stamp, load from measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = u.stamp

        if dt is None:
            dt = u.stamp - x.state.stamp

        if dt < 0:
            raise RuntimeError("dt is negative!")

        # Load dedicated jacobian evaluation point if user specified.
        if x_jac is None:
            x_jac = x.state

        if u is not None:
            A = self.process_model.jacobian(x_jac, u, dt)
            Q = self.process_model.covariance(x_jac, u, dt)
            x.state = self.process_model.evaluate(x.state, u, dt)
            x.covariance = A @ x.covariance @ A.T + Q
            x.symmetrize()
            x.state.stamp += dt

        return x

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
        output_details : bool, optional
            Whether to output intermediate computation results (innovation, innovation covariance)
                in an additional returned dict.
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
                raise RuntimeError("Measurement stamp is earlier than state stamp")
            elif u is not None:
                x = self.predict(x, u, dt)

        if x_jac is None:
            x_jac = x.state
        P = x.covariance
        R = np.atleast_2d(y.model.covariance(x_jac))
        G = np.atleast_2d(y.model.jacobian(x_jac))
        y_check = y.model.evaluate(x.state)

        if y_check is not None:
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
                x.state = x.state.plus(dx)
                x.covariance = (np.identity(x.state.dof) - K @ G) @ P
                x.symmetrize()

        details_dict = {"z": z, "S": S}
        if output_details:
            return x, details_dict
        else:
            return x


class IteratedKalmanFilter(ExtendedKalmanFilter):
    """
    On-manifold iterated extended Kalman filter.
    """

    __slots__ = ["process_model", "reject_outliers", "step_tol", "max_iters"]

    def __init__(
        self,
        process_model: ProcessModel,
        step_tol=1e-4,
        max_iters=200,  # TODO. implement max iters
        line_search=True,  # TODO implement line search
        reject_outliers=False,
    ):
        super(IteratedKalmanFilter, self).__init__(process_model)
        self.step_tol = step_tol
        self.max_iters = max_iters
        self.reject_outliers = reject_outliers

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
                raise RuntimeError("Measurement stamp is earlier than state stamp")
            elif dt > 0 and u is not None:
                x = self.predict(x, u, dt)

        dx = np.zeros((x.state.dof,))
        x_op = x.state.copy()  # Operating point
        count = 0
        while count < self.max_iters:

            # Load a dedicated state evaluation point for jacobian
            # if the user supplied it.

            info = self._get_cost_and_info(x_op, x, y, x_jac)
            J = info["J"]
            G = info["G"]
            R = info["R"]
            P = info["P"]
            e = info["e"]
            z = info["z"]
            cost_old = info["cost"]

            S = G @ P @ G.T + R
            S = 0.5 * (S + S.T)

            # Test for outlier if requested.
            outlier = False
            if reject_outlier:
                outlier = check_outlier(z, S)

            # If not outlier, compute step direction
            # Otherwise, exit loop.
            if not outlier:
                K = np.linalg.solve(S.T, (P @ G.T).T).T
                dx = -J @ e + K @ (z + G @ J @ e)
            else:
                break

            # If step direction is small already, exit loop.
            if np.linalg.norm(dx) < self.step_tol:
                break

            # Perform backtracking line search
            alpha = 1
            cost_new = cost_old + 9999
            step_accepted = False
            while not step_accepted and alpha > self.step_tol:
                x_new = x_op.plus(alpha * dx)
                cost_new = self._get_cost_and_info(x_new, x, y, x_jac)["cost"]
                if cost_new < cost_old:
                    step_accepted = True
                else:
                    alpha *= 0.9

            # If step was not accepted, exit loop and do not update step
            if not step_accepted:
                break

            x_op = x_new
            count += 1

        x.state = x_op
        if not outlier:
            x.covariance = (np.identity(x.state.dof) - K @ G) @ P

        x.symmetrize()

        return x

    def _get_cost_and_info(
        self,
        x_op: State,
        x_check: StateWithCovariance,
        y: Measurement,
        x_jac,
    ) -> float:

        if x_jac is not None:
            x_op_jac = x_jac
        else:
            x_op_jac = x_op
        R = np.atleast_2d(y.model.covariance(x_op_jac))
        G = np.atleast_2d(y.model.jacobian(x_op_jac))
        y_check = y.model.evaluate(x_op)
        z = y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))
        e = x_op.minus(x_check.state).reshape((-1, 1))
        J = x_op.jacobian(e)
        P = J @ x_check.covariance @ J.T
        P = 0.5 * (P + P.T)
        cost_prior = np.ndarray.item(0.5 * e.T @ np.linalg.solve(P, e))
        cost_meas = np.ndarray.item(0.5 * z.T @ np.linalg.solve(R, z))

        out = {
            "cost": cost_prior + cost_meas,
            "z": z,
            "G": G,
            "J": J,
            "P": P,
            "e": e,
            "R": R,
        }
        return out


def generate_sigmapoints(n: int, method: str):

    if method == "unscented":
        kappa = 2
        sigma_points = np.sqrt(n + kappa) * np.block([[np.eye(n), -np.eye(n)]])

        w_m = 1 / (2 * (n + kappa)) * np.ones((2 * n + 1))
        w_c = 1 / (2 * (n + kappa)) * np.ones((2 * n + 1))
        w_m[0] = kappa / (n + kappa)
        w_c[0] = kappa / (n + kappa) 

    elif method == "cubature":

        sigma_points = np.sqrt(n) * np.block([[np.eye(n), -np.eye(n)]])

        w_m = 1 / (2 * n) * np.ones((2 * n))
        w_c = 1 / (2 * n) * np.ones((2 * n))

    elif method == "gh":
        p = 4

        c = np.zeros(p + 1)
        c[-1] = 1
        sigma_points_scalar = hermeroots(c)
        weights_scalar = np.zeros(p)
        for i in range(1, p + 1):
            weights_scalar[i - 1] = (
                factorial(p)
                / (p * eval_hermitenorm(sigma_points_scalar[i - 1], p)) ** 2
            )
        

    return sigma_points, w_m, w_c


class SigmaPointKalmanFilter:
    """
    On-manifold nonlinear Sigma Point Kalman filter.
    """

    __slots__ = ["process_model", "reject_outliers"]

    def __init__(self, process_model: ProcessModel, method: str, reject_outliers=False):
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
        """
        self.process_model = process_model
        self.method = method
        self.reject_outliers = reject_outliers

    def predict(
        self,
        x: StateWithCovariance,
        u: StampedValue,
        input_covariance : np.ndarray,
        dt: float = None,
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
        u : StampedValue
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, dt will be calculated
            with `dt = u.stamp - x.state.stamp`.

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
            input_covariance = u.covariance

        if dt is None:
            dt = u.stamp - x.state.stamp

        if dt < 0:
            raise RuntimeError("dt is negative!")

        # Load dedicated jacobian evaluation point if user specified.
        if x_jac is None:
            x_jac = x.state

        if u is not None:

            
            n_x = x.state.dof
            n_u = u.value.size
            P = np.block(
                [[x.covariance, np.zeros((n_x, n_u))], 
                [np.zeros((n_x, n_u)), input_covariance]]
            )
            P_sqrt = np.linalg.cholesky(P)

            unit_sigmapoints, w_mean, w_cov = generate_sigmapoints(
                n_x + n_u, self.method
            )

            sigmapoints = P_sqrt @ unit_sigmapoints

            n_sig = w_mean.size
            x_propagated = np.zeros(( n_sig, x.state.value.shape))

            # Propagate
            for i in range(n_sig):
                x_propagated[i] = self.process_model.evaluate(x.state.plus(sigmapoints[0:n_x, i]),
                                                            u.plus(sigmapoints[n_x::]),
                                                            dt)
                

            # Compute mean
            #TODO Improve mean computation
            x_mean = self.process_model.evaluate(x.state,
                                                            u,
                                                            dt)


            # Compute covariance
            P_new = np.zeros(x.covariance.shape)
            for i in range(n_sig):
                err = x_mean.minus(x_propagated[i])
                err.reshape(-1,1)
                P_new += w_cov[i]* err @ err.T

            x.state = x_mean
            x.covariance = P_new
            x.symmetrize()
            x.state.stamp += dt

        return x

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        u: StampedValue,
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
                raise RuntimeError("Measurement stamp is earlier than state stamp")
            elif u is not None:
                x = self.predict(x, u, dt)


        P = x.covariance
        R = y.model.covariance(x.state)
        n_x = x.state.dof
        n_y = y.value.size
        unit_sigmapoints, w_mean, w_cov = generate_sigmapoints(
            n_x, self.method
        )

        P_sqrt = np.linalg.cholesky(P)
        sigmapoints = P_sqrt @ unit_sigmapoints

        n_sig = w_mean.size



        y_check = y.model.evaluate(x.state)

        if y_check is not None:

            
            

            y_propagated = np.zeros((n_sig, y.value.shape))

            # Propagate
            for i in range(n_sig):
                y_propagated[i] = y.model.evaluate(x.state.plus(sigmapoints[0:n_x, i]))
                                    
            #mean
            y_mean = np.zeros(y.value.shape)
            for i in range(n_sig):
                y_mean += w_mean[i]* y_propagated[i]

            #covariance innovation
            Pyy = R.copy()
            Pxy = np.zeros((n_x, n_y))
            for i in range(n_sig):
                err = y_propagated[i] - y_mean
                Pyy += w_cov[i] * err @err.T
                Pxy += w_cov[i] * sigmapoints[i] @err.T



            z = y.value.reshape((-1, 1)) - y_mean.reshape((-1, 1))
            

            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, Pyy)

            if not outlier:

                # Do the correction
                K = np.linalg.solve(Pyy.T, Pxy.T).T
                dx = K @ z
                x.state = x.state.plus(dx)
                x.covariance = P - K @ Pxy.T
                x.symmetrize()

        return x


def run_filter(
    filter: ExtendedKalmanFilter,
    x0: State,
    P0: np.ndarray,
    input_data: List[Input],
    meas_data: List[Measurement],
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
    for k in range(len(input_data) - 1):
        results_list.append(x)

        u = input_data[k]

        # Fuse any measurements that have occurred.
        if len(meas_data) > 0:
            while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

                x = filter.correct(x, y, u)
                meas_idx += 1
                if meas_idx < len(meas_data):
                    y = meas_data[meas_idx]

        dt = input_data[k + 1].stamp - x.stamp
        x = filter.predict(x, u, dt)

    return results_list
