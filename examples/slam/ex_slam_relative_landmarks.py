"""This example will show a simple SLAM example where a robot
equipped with an IMU moves in 3D space and receives relative 
position measurements of landmarks with unknown locations.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from pylie import SE23
from pynav.slam import ExtendedKalmanFilterSLAM
from pynav.filters import run_filter
from pynav.lib.models import (
    PointRelativePosition,
    PointRelativePositionSLAM,
)
from pynav.lib.states import CompositeState, VectorState
from pynav.lib.imu import IMU, IMUState, IMUKinematics
from pynav.types import StateWithCovariance
from pynav.utils import GaussianResult, GaussianResultList, plot_error, randvec
from pynav.datagen import DataGenerator
from slam_datagen import generate_landmark_positions

# ##############################################################################
# Problem Setup

t_start = 0
t_end = 30
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

# Initial navigation state
nav_state_init = SE23.from_components(
    np.identity(3),
    np.array([3, 0, 0]).reshape((-1, 1)),
    np.array([0, 3, 0]).reshape((-1, 1)),
)

# Continuous-time Power Spectral Density
Q_c = np.eye(12)
Q_c[0:3, 0:3] *= sigma_gyro_ct**2
Q_c[3:6, 3:6] *= sigma_accel_ct**2
Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2
dt = 1 / imu_freq
Q_noise = Q_c / dt

# Generate landmark positions
landmarks = generate_landmark_positions(
    cylinder_radius, max_height, n_level, n_landmarks_per_level
)

# Create process model and measurement model
process_model = IMUKinematics(Q_c / dt)
meas_cov = np.identity(3) * sigma_landmark_sensor**2
meas_model_list = [
    PointRelativePosition(pos, meas_cov, "l" + str(i))
    for i, pos in enumerate(landmarks)
]


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
    bias_noise = randvec(Q_bias)

    u = IMU(omega, accel, stamp, bias_noise[0:3], bias_noise[3:6])
    return u


# Create data generator
data_gen = DataGenerator(
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
    state_id="r",
    direction="left",
)

P0 = 1e-7 * np.eye(15)

# Use datagen to generate the IMU states
true_imu_states, input_list, meas_list = data_gen.generate(
    x0, t_start, t_end, noise=True
)

# Zero-out the random walk values
for u in input_list:
    u.bias_gyro_walk = np.array([0, 0, 0])
    u.bias_accel_walk = np.array([0, 0, 0])

# Create a SLAM state that contains
# both the landmark states and the robot state.
x0 = true_imu_states[0].copy()
state_list = [x0]

for i, pos in enumerate(landmarks):
    landmark_state = VectorState(pos, 0.0, "l" + str(i))
    state_list.append(landmark_state)

slam_state = CompositeState(state_list, stamp=0.0)

# Convert measurements to SLAM measurements
slam_meas_list = []
for meas in meas_list:
    landmark_id = meas.model.landmark_id
    meas.model = PointRelativePositionSLAM(landmark_id, meas.model._R)
    # Each measurement is a function of the robot state and one landmark.
    meas.state_id = [x0.state_id, landmark_id]


# Augment covariance with landmark estimates
imu_dof = slam_state.value[0].dof
init_cov = 1e-8 * np.identity(slam_state.dof)

# Create ExtendedKalmanFilterSLAM
ekf_slam = ExtendedKalmanFilterSLAM(process_model)

# Run the filter on the data
estimate_list = run_filter(
    ekf_slam, slam_state, init_cov, input_list, meas_list
)

# Extract all IMU states
imu_estimates = []
for estimate in estimate_list:
    imu_state = estimate.state.value[0]
    imu_cov = estimate.covariance[0:15, 0:15]
    imu_estimates.append(StateWithCovariance(imu_state, imu_cov))

# Postprocess the results and plot
results = GaussianResultList(
    [
        GaussianResult(imu_estimates[i], true_imu_states[i])
        for i in range(len(imu_estimates))
    ]
)

from pynav.utils import plot_poses
import seaborn as sns

fig = plt.figure()
ax = plt.axes(projection="3d")
landmarks = np.array(landmarks)
ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
states_list = [x.state for x in imu_estimates]
plot_poses(states_list, ax, line_color="tab:blue", step=500, label="Estimate")
plot_poses(
    true_imu_states, ax, line_color="tab:red", step=500, label="Groundtruth"
)
ax.legend()

sns.set_theme()
fig, axs = plot_error(results)
axs[0, 0].set_title("Attitude")
axs[0, 1].set_title("Velocity")
axs[0, 2].set_title("Position")
axs[0, 3].set_title("Gyro bias")
axs[0, 4].set_title("Accel bias")
axs[-1, 2]

plt.show()
