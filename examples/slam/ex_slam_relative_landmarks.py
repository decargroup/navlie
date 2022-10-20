"""This example will show a simple SLAM example where a robot
equipped with an IMU moves in 3D space and receives relative 
position measurements of landmarks with unknown locations.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from pylie import SE23
from pynav.filters import (
    ExtendedKalmanFilter,
    ExtendedKalmanFilterSLAM,
    run_filter,
    run_slam_ekf,
)
from pynav.lib.models import (
    InvariantMeasurement,
    PointRelativePosition,
    PointRelativePositionSLAM,
)
from pynav.lib.states import CompositeState, VectorState
from pynav.lib.imu import IMU, IMUState, IMUKinematics
from pynav.utils import GaussianResult, GaussianResultList, plot_error, randvec
from pynav.datagen import DataGenerator
from slam_datagen import generate_landmark_positions

# ##############################################################################
# Problem Setup

t_start = 0
t_end = 10
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

landmarks = generate_landmark_positions(
    cylinder_radius, max_height, n_level, n_landmarks_per_level
)

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
    state_id=0,
    direction="left",
)

P0 = np.eye(15)
P0[0:3, 0:3] *= sigma_phi_init**2
P0[3:6, 3:6] *= sigma_v_init**2
P0[6:9, 6:9] *= sigma_r_init**2
P0[9:12, 9:12] *= sigma_bg_init**2
P0[12:15, 12:15] *= sigma_ba_init**2

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
x0.state_id = "p0"
state_list = [x0]

for i, pos in enumerate(landmarks):
    landmark_state = VectorState(pos, 0.0, "l" + str(i))
    state_list.append(landmark_state)

slam_state = CompositeState(state_list, stamp=0.0)

# Convert measurements to SLAM measurements
slam_meas_list = []
for meas in meas_list:
    new_model = PointRelativePositionSLAM(meas.model.landmark_id, meas.model._R)
    meas.model = new_model

# Augment covariance with landmark estimates
imu_dof = slam_state.value[0].dof
init_cov = np.zeros((slam_state.dof, slam_state.dof))
init_cov[:imu_dof, :imu_dof] = P0
init_cov[imu_dof:, imu_dof:] = 1e-7 * np.identity(slam_state.dof - imu_dof)

# Create ExtendedKalmanFilterSLAM
ekf_slam = ExtendedKalmanFilterSLAM(process_model)

# Run the filter on the data
estimate_list = run_slam_ekf(
    ekf_slam, slam_state, init_cov, input_list, meas_list
)

# Postprocess the results and plot
results = GaussianResultList(
    [
        GaussianResult(estimate_list[i], states_true[i])
        for i in range(len(estimate_list))
    ]
)
