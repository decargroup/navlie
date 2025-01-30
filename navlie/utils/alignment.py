"""
Utilities for interfacing with the evo package for trajectory alignment and
evaluation.
"""

import numpy as np
import typing
from pymlg import SE3, SO3
from navlie.lib.states import SE3State

from evo.core import trajectory
from evo.core import sync


def state_list_to_evo_traj(
    state_list: typing.List[SE3State],
) -> trajectory.PoseTrajectory3D:
    """Converts a list of SE3States to an evo trajectory.

    Parameters
    ----------
    state_list : typing.List[SE3State]
        The list of SE3States to convert

    Returns
    -------
    trajectory.PoseTrajectory3D
        The converted evo trajectory
    """
    positions = np.array([state.position for state in state_list])
    quats = np.array(
        [SO3.to_quat(x.attitude, order="wxyz").ravel() for x in state_list]
    )
    stamps = np.array([x.stamp for x in state_list])
    traj = trajectory.PoseTrajectory3D(positions, quats, stamps)
    return traj


def evo_traj_to_state_list(
    traj: trajectory.PoseTrajectory3D,
) -> typing.List[SE3State]:
    """Converts an evo trajectory to a list of SE3State.

    Parameters
    ----------
    traj : trajectory.PoseTrajectory3D
        The evo pose trajectory to convert

    Returns
    -------
    typing.List[SE3State]
        The converted list of SE3State
    """
    state_list = []
    for i in range(len(traj.timestamps)):
        pose_mat = SE3.from_components(
            SO3.from_quat(traj.orientations_quat_wxyz[i, :], order="wxyz"),
            traj._positions_xyz[i, :],
        )
        state = SE3State(
            value=pose_mat,
            stamp=traj.timestamps[i],
        )
        state.stamp = traj.timestamps[i]
        state_list.append(state)
    return state_list


def associate_and_align_trajectories(
    traj_ref_list: typing.List[SE3State],
    traj_est_list: typing.List[SE3State],
    max_diff: float = 0.02,
    offset: float = 0.0,
    align: bool = True,
    correct_scale: bool = False,
    n_to_align: int = -1,
    verbose: bool = False,
) -> typing.Tuple[
    typing.List[SE3State], typing.List[SE3State], typing.Dict[str, np.ndarray]
]:
    """Associates the stamps of two trajectories and aligns the
    estimated trajectory to the reference trajectory.

    Parameters
    ----------
    traj_ref : typing.List[SE3State]
        Reference trajectory, a list of SE3State
    traj_est : typing.List[SE3State]
        Estimated trajectory to be aligned to the reference, a list of SE3State
    max_diff : float, optional
        The maximum allowable difference in timestamps between the estimate
        and the reference trajectory, by default 0.02
    offset : float, optional
        optional time offset of the second trajectory, by default 0.0
    align: bool, optional
        Whether to align the trajectories, by default True
    correct_scale: bool, optional
        Whether to correct the scale, by default False. If correct_scale is
        false, the alignment transformation is an element of SE(3), otherwise,
        it is an element of Sim(3).
    n_to_align: int, optional
        The number of poses to use for alignment, by default -1. If n_to_align
        is -1, all poses are used.
    verbose : bool, optional
        Verbosity flag, by default False

    Returns
    -------
    typing.List[SE3State]
        The reference trajectory, with timestamps at the same times as
        the estimated trajectory.
    typing.List[SE3State]
        The aligned estimated trajectory.
    typing.Dict[str, np.ndarray]
        A dictionary containing the transformation information. The keys are:
        - "rotation": The DCM of the alignment transformation
        - "position": The transformation component of the alignment transformation
        - "scale": The scale component alignment transformation.
    """
    # Convert trajectories to evo, and associate the timestamps
    traj_ref = state_list_to_evo_traj(traj_ref_list)
    traj_est = state_list_to_evo_traj(traj_est_list)
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_diff,
        offset_2=offset,
    )

    if verbose:
        print(f"Number of matching states: {len(traj_ref_sync.timestamps)}")
        print(f"Length of reference trajectory: {len(traj_ref_list)}")
        print(f"Length of estimated trajectory: {len(traj_est_list)}")

    # If we are not aligning, return the synced trajectories
    if not align:
        traj_ref_sync = evo_traj_to_state_list(traj_ref_sync)
        traj_est_sync = evo_traj_to_state_list(traj_est_sync)
        transformation_dict = {
            "rotation": np.identity(3),
            "position": np.zeros(3),
            "scale": 1.0,
        }
        return traj_ref_sync, traj_est_sync, transformation_dict

    # Otherwise, align the trajectories
    traj_est_sync: trajectory.PoseTrajectory3D
    R, t, s = traj_est_sync.align(
        traj_ref_sync,
        correct_scale=correct_scale,
        n=n_to_align,
    )

    # Conver back to navlie types
    traj_est_aligned = evo_traj_to_state_list(traj_est_sync)
    traj_ref_aligned = evo_traj_to_state_list(traj_ref_sync)
    transformation_dict = {
        "rotation": R,
        "position": t,
        "scale": s,
    }

    return traj_ref_aligned, traj_est_aligned, transformation_dict
