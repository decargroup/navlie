""" 
A slightly more complicated example of a robot localizing itself from relative
position measurements to known landmarks.
"""
from typing import List
import numpy as np
from pymlg import SE23
from navlie.lib import (
    IMU,
    IMUState,
    IMUKinematics,
    InvariantMeasurement,
    PointRelativePosition,
)
import navlie as nav

# ##############################################################################
# Problem Setup

t_start = 0
t_end = 15
imu_freq = 100

# IMU noise parameters
sigma_gyro_ct = 0.01
sigma_accel_ct = 0.01
sigma_gyro_bias_ct = 0.0001
sigma_accel_bias_ct = 0.0001
init_gyro_bias = np.array([0.02, 0.03, -0.04]).reshape((-1, 1))
init_accel_bias = np.array([0.01, 0.02, 0.05]).reshape((-1, 1))

# Landmark parameters
cylinder_radius = 7
n_level = 1
n_landmarks_per_level = 3
max_height = 2.5

# Landmark sensor parameters
sigma_landmark_sensor = 0.1  # [m]
landmark_sensor_freq = 10

# Initialization parameters
sigma_phi_init = 0.1
sigma_v_init = 0.1
sigma_r_init = 0.1
sigma_bg_init = 0.01
sigma_ba_init = 0.01
nav_state_init = SE23.from_components(
    np.identity(3),
    np.array([3, 0, 0]).reshape((-1, 1)),
    np.array([0, 3, 0]).reshape((-1, 1)),
)

################################################################################
################################################################################

# Continuous-time Power Spectral Density
Q_c = np.eye(12)
Q_c[0:3, 0:3] *= sigma_gyro_ct**2
Q_c[3:6, 3:6] *= sigma_accel_ct**2
Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2
dt = 1 / imu_freq

Q_noise = Q_c / dt


def generate_landmark_positions(
    cylinder_radius: float,
    max_height: float,
    n_levels: int,
    n_landmarks_per_level: int,
) -> List[np.ndarray]:
    """Generates landmarks arranged in a cylinder.

    Parameters
    ----------
    cylinder_radius : float
        Radius of the cylinder that the landmarks are arranged in.
    max_height : float
        Top of cylinder.
    n_levels : int
        Number of discrete levels to place landmarks at vertically.
    n_landmarks_per_level : int
        Number of landmarks per level

    Returns
    -------
    List[np.ndarray]
        List of landmarks.
    """

    z = np.linspace(0, max_height, n_levels)

    angles = np.linspace(0, 2 * np.pi, n_landmarks_per_level + 1)
    angles = angles[0:-1]
    x = cylinder_radius * np.cos(angles)
    y = cylinder_radius * np.sin(angles)

    # Generate landmarks
    landmarks = []
    for level_idx in range(n_levels):
        for landmark_idx in range(n_landmarks_per_level):
            cur_landmark = np.array(
                [x[landmark_idx], y[landmark_idx], z[level_idx]]
            ).reshape((3, -1))
            landmarks.append(cur_landmark)

    return landmarks


def input_profile(stamp: float, x: IMUState) -> np.ndarray:
    """Generates an IMU measurement for a circular trajectory,
    where the robot only rotates about the z-axis and the acceleration
    points towards the center of the circle.
    """

    # Add biases to true angular velocity and acceleration
    bias_gyro = x.bias_gyro.reshape((-1, 1))
    bias_accel = x.bias_accel.reshape((-1, 1))

    C_ab = x.attitude
    g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))
    omega = np.array([0, 0, -1.0]).reshape((-1, 1)) + bias_gyro
    accel = np.array([0, -3.0, 0]).reshape((-1, 1)) + bias_accel - C_ab.T @ g_a

    # Generate a random input to drive the bias random walk
    Q_bias = Q_noise[6:, 6:]
    bias_noise = nav.randvec(Q_bias)

    u = IMU(omega, accel, stamp, bias_noise[0:3], bias_noise[3:6])
    return u


landmarks = generate_landmark_positions(
    cylinder_radius, max_height, n_level, n_landmarks_per_level
)

process_model = IMUKinematics(Q_c / dt)
meas_cov = np.identity(3) * sigma_landmark_sensor**2
meas_model_list = [PointRelativePosition(pos, meas_cov) for pos in landmarks]

# Create data generator
data_gen = nav.DataGenerator(
    process_model,
    input_func=input_profile,
    input_covariance=Q_noise,
    input_freq=imu_freq,
    meas_model_list=meas_model_list,
    meas_freq_list=landmark_sensor_freq,
)

# Initial state and covariance
x0 = IMUState(
    nav_state_init,
    init_gyro_bias,
    init_accel_bias,
    stamp=t_start,
    state_id=0,
    direction="left",
)

P0 = np.eye(15)
P0[0:3, 0:3] *= sigma_phi_init**2
P0[3:6, 3:6] *= sigma_v_init**2
P0[6:9, 6:9] *= sigma_r_init**2
P0[9:12, 9:12] *= sigma_bg_init**2
P0[12:15, 12:15] *= sigma_ba_init**2

# Generate all data
states_true, input_list, meas_list = data_gen.generate(x0, t_start, t_end, noise=True)

# **************** Conversion to Invariant Measurements ! *********************
invariants = [InvariantMeasurement(meas, "right") for meas in meas_list]
# *****************************************************************************

# Zero-out the random walk values
for u in input_list:
    u.bias_gyro_walk = np.array([0, 0, 0])
    u.bias_accel_walk = np.array([0, 0, 0])


ekf = nav.ExtendedKalmanFilter(process_model)

estimate_list = nav.run_filter(ekf, x0, P0, input_list, invariants)

# Postprocess the results and plot
results = nav.GaussianResultList.from_estimates(estimate_list, states_true)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    landmarks = np.array(landmarks)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
    states_list = [x.state for x in estimate_list]
    nav.plot_poses(results.state, ax, line_color="tab:blue", step=500, label="Estimate")
    nav.plot_poses(
        results.state_true, ax, line_color="tab:red", step=500, label="Groundtruth"
    )
    ax.legend()

    sns.set_theme()
    fig, axs = nav.plot_error(results)
    axs[0, 0].set_title("Attitude")
    axs[0, 1].set_title("Velocity")
    axs[0, 2].set_title("Position")
    axs[0, 3].set_title("Gyro bias")
    axs[0, 4].set_title("Accel bias")
    axs[-1, 2]

    plt.show()
