"""This script will show how to use the B-Spline simulator to simulate robot
trajectories in 3D space, and to generate measurements along the trajectory
to then run estimation algorithms on.

The script simulates a trajectory based on a sequence from the Euroc dataset,
where the groundtruth file is provided under navlie/data/MH_01_easy.txt. 

The B-Spline simulator is used to generate IMU measurements and absolute
position measurements along the trajectory, and then an EKF is used to fuse them. 
"""

import os
import numpy as np
import typing

from pymlg import SE23
from navlie.lib.states import SE3State
from navlie.bspline import SE3Bspline
from navlie.utils.common import load_tum_trajectory, randvec
from navlie.lib.imu import IMU, IMUState, IMUKinematics
from navlie.lib.models import AbsolutePosition
from navlie.types import Measurement
import navlie as nav

import matplotlib.pyplot as plt

cur_dir = os.path.dirname(os.path.realpath(__file__))


def generate_bspline_dataset(
    traj_file: str,
    input_freq: int = 200,
    meas_freq: int = 5,
    t_end: float = 100,
    Q_c: np.ndarray = None,
    R: np.ndarray = None,
    gravity_mag: float = 9.80665,
) -> typing.Dict[str, typing.Any]:
    """This function generates simulated trajectory using the
    SE(3) B-Spline simulator found in the module bspline.py.

    In addition, this function generates simulated noisy, biased
    IMU measurements and noisy GPS measurements along the trajectory.

    Parameters
    ----------
    traj_file : str
        The path to the TUM trajectory file to load
    input_freq : int, optional
        The frequency of the IMU measurements, by default 200.
    meas_freq : int, optional
        The frequency of the GPS position measurements, by default 5.
    t_end : float, optional
        The duration of the trajectory to simulate, by default 100.
    Q_c : np.ndarray, optional
        The continuous-time IMU process noise covariance, by default None.
        This should be a 12x12 matrix, where the ordering is
            gyro_white_noise, accel_white_noise,
            gyro_bias_white_noise, accel_bias_white_noise.
    R : np.ndarray, optional
        The measurement noise covariance for the GPS measurements, by default None.
    gravity_mag : float, optional
        The magnitude of the gravity vector, by default 9.80665.

    Returns:
        typing.Dict[str, typing.Any]: A dictionary containing the following
        keys:
            - gt_states: A list of the groundtruth IMU states
            - imu_meas: A list of the noisy IMU measurements
            - abs_pos_meas: A list of the noisy GPS measurements
            - process_noise: The discrete-time process noise covariance
    """
    # Load the trajectory until the end time
    raw_traj = load_tum_trajectory(traj_file)
    start_stamp = raw_traj[0].stamp
    end_stamp = start_stamp + t_end

    trajectory = []
    for pose in raw_traj:
        if pose.stamp < end_stamp:
            trajectory.append(pose)
        else:
            break

    # Create the B-Spline from the raw trajectory
    bspline = SE3Bspline(trajectory, verbose=True)

    # Query the B-Spline to get the poses and velocities at a set of times
    dt = 1.0 / input_freq
    t_eval = np.arange(bspline.start_time, bspline.end_time, dt)
    poses_gt: typing.List[SE3State] = []
    angular_vels: typing.List[np.ndarray] = []
    linear_vels: typing.List[np.ndarray] = []
    linear_accels: typing.List[np.ndarray] = []
    stamps: typing.List[float] = []
    for t in t_eval:
        pose = bspline.get_pose(t)
        omega_b_ba, v_a_ba = bspline.get_velocity(t)
        _, a_a_ba = bspline.get_acceleration(t)

        if pose is not None and omega_b_ba is not None:
            poses_gt.append(pose)
            angular_vels.append(omega_b_ba)
            linear_vels.append(v_a_ba)
            linear_accels.append(a_a_ba)
            stamps.append(t)

    # Set the continuous-time IMU noise covariance
    if Q_c is None:
        Q_c = np.identity(12)
        Q_c[0:3, 0:3] = np.identity(3) * 0.0001
        Q_c[3:6, 3:6] = np.identity(3) * 0.001
        Q_c[6:9, 6:9] = np.identity(3) * 0.0001
        Q_c[9:, 9:] = np.identity(3) * 0.0001
    if R is None:
        R = np.identity(3) * 0.01
    # Convert the continuous-time noise to discrete-time
    Q_d = Q_c / dt

    # Extract the relevant subblocks of the process noise to generate
    # the measurements
    Qd_gyro = Q_d[0:3, 0:3]
    Qd_accel = Q_d[3:6, 3:6]
    Qd_gyro_bias = Q_d[6:9, 6:9]
    Qd_accel_bias = Q_d[9:, 9:]

    # Propagate forward the IMU biases using a first order random walk
    init_gyro_bias = np.array([0.02, 0.03, -0.03])
    init_accel_bias = np.array([0.02, -0.06, 0.04])
    gyro_biases: typing.List[np.ndarray] = [init_gyro_bias]
    accel_biases: typing.List[np.ndarray] = [init_accel_bias]
    for i in range(0, len(poses_gt) - 1):
        next_gyro_bias = gyro_biases[i] + dt * randvec(Qd_gyro_bias).ravel()
        next_accel_bias = accel_biases[i] + dt * randvec(Qd_accel_bias).ravel()
        gyro_biases.append(next_gyro_bias)
        accel_biases.append(next_accel_bias)

    # Create a list of the groundtruth states
    gt_imu_states: typing.List[IMUState] = []
    for i in range(len(poses_gt)):
        pose_gt = poses_gt[i]
        vel_gt = linear_vels[i]
        gyro_bias_gt = gyro_biases[i]
        accel_bias_gt = accel_biases[i]
        stamp = stamps[i]
        nav_state = SE23.from_components(pose_gt.attitude, vel_gt, pose_gt.position)
        imu_state = IMUState(
            nav_state=nav_state,
            bias_gyro=gyro_bias_gt,
            bias_accel=accel_bias_gt,
            stamp=stamp,
        )
        gt_imu_states.append(imu_state)

    # Generate the noisy, biases IMU measurements
    # Note that the gyro measurements are of the form
    #   u_gyro = omega_b_ba + bias_gyro + noise,
    # and the accel measurements are of the form
    #   u_accel = C_ab.T * (accel_a - gravity) + bias_accel + noise
    gravity = np.array([0, 0, -gravity_mag])
    imu_meas_list: typing.List[IMU] = []
    for i in range(len(gt_imu_states)):
        cur_state = gt_imu_states[i]
        omega_b_ba = angular_vels[i].ravel()
        accel_a = linear_accels[i].ravel()
        gyro_meas = omega_b_ba + cur_state.bias_gyro + randvec(Qd_gyro).ravel()
        accel_meas = (
            cur_state.attitude.T @ (accel_a - gravity)
            + cur_state.bias_accel
            + randvec(Qd_accel).ravel()
        )
        imu_meas = IMU(
            gyro=gyro_meas,
            accel=accel_meas,
            stamp=cur_state.stamp,
        )
        imu_meas_list.append(imu_meas)

    # Generate the noisy absolute position measurements at the correct timestamps
    time_last_meas = stamps[0]
    meas_dt = 1 / meas_freq
    meas_model = AbsolutePosition(R)
    meas_list: typing.List[Measurement] = []
    for i in range(len(gt_imu_states)):
        cur_stamp = gt_imu_states[i].stamp
        if (cur_stamp - time_last_meas) > meas_dt:
            # Generate a measurement
            time_last_meas = cur_stamp
            pos_meas = gt_imu_states[i].position + randvec(R).ravel()
            meas = Measurement(
                model=meas_model,
                value=pos_meas,
                stamp=cur_stamp,
            )
            meas_list.append(meas)

    # Return the dataset
    out_dict = {
        "gt_states": gt_imu_states,
        "imu_meas": imu_meas_list,
        "abs_pos_meas": meas_list,
        "process_noise": Q_d,
    }
    return out_dict


def main(traj_file: str) -> nav.GaussianResultList:
    # Generate the simualted dataset
    dataset = generate_bspline_dataset(
        traj_file,
        input_freq=200,
        meas_freq=10,
        t_end=300,
    )

    # Plot the IMU measurements along the trajectory 
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    stamps = np.array([m.stamp for m in dataset["imu_meas"]])
    gyro_meas = np.array([m.gyro for m in dataset["imu_meas"]])
    accel_meas = np.array([m.accel for m in dataset["imu_meas"]])
    ax[0].plot(stamps, gyro_meas, label="Gyro Meas")
    ax[0].set_ylabel("Gyro Meas (rad/s)")
    ax[1].plot(stamps, accel_meas, label="Accel Meas")
    ax[1].set_ylabel("Accel Meas (m/s^2)")
    ax[1].set_xlabel("Time (s)")
    fig.tight_layout()

    # Run a filter on the data!
    gt_states = dataset["gt_states"]
    imu_meas = dataset["imu_meas"]
    abs_pos_meas = dataset["abs_pos_meas"]
    Q_d = dataset["process_noise"]

    # Create the process model
    process_model = IMUKinematics(Q_d)

    # Filter initialization
    P0 = np.eye(15)
    P0[0:3, 0:3] *= 0.01**2
    P0[3:6, 3:6] *= 0.01**2
    P0[6:9, 6:9] *= 0.01**2
    P0[9:12, 9:12] *= 0.01**2
    P0[12:15, 12:15] *= 0.01**2
    x0 = gt_states[0].plus(randvec(P0))

    # ###########################################################################
    # Run filter
    ekf = nav.ExtendedKalmanFilter(process_model)
    estimate_list = nav.run_filter(ekf, x0, P0, imu_meas, abs_pos_meas)
    return nav.GaussianResultList.from_estimates(estimate_list, gt_states)


if __name__ == "__main__":
    # Select the trajecotry file to use
    traj_file = os.path.join(cur_dir, "../data", "MH_01_easy.txt")
    results = main(traj_file)
    # Plot results
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    nav.plot_poses(
        results.state,
        ax,
        line_color="tab:blue",
        step=None,
        label="Estimate",
    )
    nav.plot_poses(
        results.state_true,
        ax,
        line_color="tab:red",
        step=None,
        label="Groundtruth",
    )
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    fig, axs = nav.plot_error(results)
    axs[0, 0].set_title("Attitude")
    axs[0, 1].set_title("Velocity")
    axs[0, 2].set_title("Position")
    axs[0, 3].set_title("Gyro bias")
    axs[0, 4].set_title("Accel bias")
    axs[-1, 2]

    plt.show()
