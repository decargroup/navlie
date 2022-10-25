from pynav.types import StateWithCovariance, Input, State, Measurement
from pynav.filters import ExtendedKalmanFilter
from typing import List, Any
import numpy as np
from pynav.lib.states import CompositeState
from pynav.filters import check_outlier
from itertools import product


def set_state_with_covariance_from_ids(
    full_state: CompositeState,
    full_cov: np.ndarray,
    sub_state: CompositeState,
    sub_covariance: np.ndarray,
    state_ids: Any,
) -> StateWithCovariance:
    """Updates a state from a substate and covariance."""

    if not isinstance(state_ids, list):
        state_ids = [state_ids]

    # Set all states.
    for state_id in state_ids:
        cur_state = sub_state.get_state_by_id(state_id)
        full_state.set_state_by_id(cur_state, state_id)

    # Get all state pairs
    pairs = list(product(state_ids, repeat=2))

    # Compute covariance block for that state pair
    for pair in pairs:
        new_cov_block = sub_state.get_matrix_block_by_ids(
            sub_covariance, pair[0], pair[1]
        )

        slice_1 = full_state.get_slice_by_id(pair[0])
        slice_2 = full_state.get_slice_by_id(pair[1])

        full_cov[slice_1, slice_2] = new_cov_block

    return StateWithCovariance(full_state, full_cov)


def create_state_with_covariance_from_ids(
    full_state: CompositeState, full_cov: np.ndarray, state_ids: Any
) -> StateWithCovariance:
    """Extract the relevant portions of a full covariance matrix
    and create a new state.

    The returned state will always be a composite state, even if only
    one state was extracted.

    Parameters
    ----------
    full_state : CompositeState
        Full state to extract substates
    full_cov : np.ndarray
        Full covariance matrix of state.
    state_ids : Any
        State IDs to extract from full states.

    Returns
    -------
    StateWithCovariance
        New state with covariance, containing states with state_id.
    """
    # Create a new composite state from given state_ids
    new_state_list = []
    for state_id in state_ids:
        new_state_list.append(full_state.get_state_by_id(state_id))
    new_state = CompositeState(new_state_list)
    new_cov = np.zeros((new_state.dof, new_state.dof))

    # Get all state pairs
    pairs = list(product(state_ids, repeat=2))

    # Compute covariance block for that state pair
    for pair in pairs:
        cov_block = full_state.get_matrix_block_by_ids(full_cov, pair[0], pair[1])
        new_cov = new_state.set_matrix_block_by_ids(
            cov_block, new_cov, pair[0], pair[1]
        )

    return StateWithCovariance(new_state, new_cov)


def extract_covariance_columns(state: CompositeState, cov: np.ndarray, state_ids: Any):
    cov_col_list = []
    for state_id in state_ids:
        slice = state.get_slice_by_id(state_id)
        cov_col_list.append(cov[:, slice])
    return np.hstack(cov_col_list)


class ExtendedKalmanFilterSLAM(ExtendedKalmanFilter):
    """A Kalman filter specifically when working with
    SLAM problems.
    """

    def predict(
        self,
        x: StateWithCovariance,
        u: Input,
        dt: float = None,
        x_jac: State = None,
    ):
        """The prediction step for EKF-SLAM.

        The particular partitioning of the covariance matrix is utilized
        and only the full EKF prediction step is used for the robot state.
        """
        x = x.copy()

        # Extract robot state and covariance and execute standard EKF prediction step
        robot_state_id = x.state.value[0].state_id
        r_dof = x.state.value[0].dof
        robot_state = StateWithCovariance(
            x.state.value[0], x.covariance[0:r_dof, 0:r_dof]
        )

        robot_state, A = super().predict(robot_state, u, dt, x_jac, output_jac=True)

        # Set the new robot state and covariance
        x.state.set_state_by_id(robot_state.state, robot_state_id)
        x.covariance[0:r_dof, 0:r_dof] = robot_state.covariance

        # Set cross-covariances between robot and map
        P_rm = x.covariance[:r_dof, r_dof:]
        x.covariance[:r_dof, r_dof:] = A @ P_rm
        x.covariance[r_dof:, :r_dof] = P_rm.T @ A.T

        x.stamp += dt
        x.symmetrize()

        return x

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        u: Input,
        x_jac: State = None,
        reject_outlier: bool = None,
        output_details: bool = False,
    ):
        """Correction step for EKF-SLAM, exploiting the sparsity of the
        problem.

        TODO: There is much code reuse between this and the standard EKF,
        refactor to reduce this.
        """
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
            elif u is not None and dt > 0:
                x = self.predict(x, u, dt)

        # Get the substate that this measurement is a function of
        sub_x = create_state_with_covariance_from_ids(x.state, x.covariance, y.state_id)

        if x_jac is None:
            x_jac = sub_x.state

        # Evaluate model covariance, Jacobian and measurement at the substate
        R = np.atleast_2d(y.model.covariance(sub_x.state))
        G = np.atleast_2d(y.model.jacobian(sub_x.state))
        y_check = y.model.evaluate(sub_x.state)

        # Compute innovation covariance
        if y_check is not None:
            z = y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))
            S = G @ sub_x.covariance @ G.T + R

            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:
                # Do the correction
                # We need to compute the portion of the covariance matrix
                # to compute the Kalman gain with

                P_cols = extract_covariance_columns(x.state, x.covariance, y.state_id)
                K = np.linalg.solve(S.T, (P_cols @ G.T).T).T
                dx = K @ z
                x.state = x.state.plus(dx)
                x.covariance = x.covariance - K @ S @ K.T
                x.symmetrize()

        details_dict = {"z": z, "S": S}
        if output_details:
            return x, details_dict
        else:
            return x
