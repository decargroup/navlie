from pynav.types import StateWithCovariance, Input, State, Measurement
from pynav.filters import ExtendedKalmanFilter
from typing import List, Any
import numpy as np
from pynav.lib.states import CompositeState


def create_state_with_covariance_from_id(
    state: CompositeState, covariance: np.ndarray, state_id: Any
):
    """From a given state with covariance, where state is a composite
    state, extract the relevant portions of the covariance matrix and create a
    new state.
    """

    new_state = state.get_state_by_id(state_id)
    new_cov = state.get_matrix_block_by_ids(covariance, state_id)

    return StateWithCovariance(new_state, new_cov)


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
        robot_state = create_state_with_covariance_from_id(
            x.state, x.covariance, robot_state_id
        )
        robot_state, A = super().predict(
            robot_state, u, dt, x_jac, output_jac=True
        )

        # Set the new robot state and covariance
        r_dof = x.state.value[0].dof
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
        # Extract portion of state that this measurement is a function of
        states_list = [x.state.get_state_by_id(id) for id in y.state_id]
        sub_state = CompositeState(states_list)

        # Also need to extract the covariance
        old_P = x.covariance
        robot_state_id = y.state_id[0]
        landmark_state_id = y.state_id[1]

        P_rr = sub_state.get_matrix_block_by_ids(old_P, robot_state_id)
        P_ll = sub_state.get_matrix_block_by_ids(old_P, landmark_state_id)
        P_rl = sub_state.get_matrix_block_by_ids(
            old_P, robot_state_id, landmark_state_id
        )

        sub_P = np.block([[P_rr, P_rl], [P_rl.T, P_ll]])
        sub_x = StateWithCovariance(sub_state, sub_P)

        # Perform correction step for this substate
        sub_x = super().correct(sub_x, y, u)
        sub_state_new: CompositeState = sub_x.state
        P_new = sub_x.covariance

        # Now, we need assign these substates back to the full state.
        P_rr = sub_state.get_matrix_block_by_ids(P_new, robot_state_id)
        P_ll = sub_state.get_matrix_block_by_ids(P_new, landmark_state_id)
        P_rl = sub_state.get_matrix_block_by_ids(
            P_new, robot_state_id, landmark_state_id
        )

        # Set all covariance blocks
        r_slice = x.state.get_slice_by_id(robot_state_id)
        l_slice = x.state.get_slice_by_id(landmark_state_id)

        x.covariance[r_slice, r_slice] = P_rr
        x.covariance[r_slice, l_slice] = P_rl
        x.covariance[l_slice, r_slice] = P_rl.T
        x.covariance[l_slice, l_slice] = P_ll

        # Set all state values
        for sub_state in sub_state_new.value:
            x.state.set_state_by_id(sub_state, sub_state.state_id)

        x.symmetrize()

        if output_details:
            return x, None
        else:
            return x
