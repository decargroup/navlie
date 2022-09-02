from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from pylie import SE23
from pynav.lib.models import (
    IMUKinematics,
    PointRelativePosition,
)
from pynav.lib.states import IMUState
from pynav.types import StampedValue, Measurement
from pynav.utils import randvec
from pynav.datagen import DataGenerator


@dataclass
class SimulationConfig:
    """Dataclass to store simulation config parameters"""

    t_start: float
    t_end: float
    input_freq: int

    # IMU noise parameters
    sigma_gyro_ct: float = 0.01
    sigma_accel_ct: float = 0.01
    sigma_gyro_bias_ct: float = 0.0001
    sigma_accel_bias_ct: float = 0.0001
    init_gyro_bias: np.ndarray = np.array([0.02, 0.03, -0.04]).reshape((3, -1))
    init_accel_bias: np.ndarray = np.array([0.01, 0.02, 0.05]).reshape((3, -1))

    # Landmark parameters
    cylinder_radius: float = 7
    n_level: int = 1
    n_landmarks_per_level: int = 3
    max_height: float = 2.5

    # Landmark sensor parameters
    sigma_landmark_sensor: float = 0.1  # [m]
    landmark_sensor_freq: int = 10


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


def input_profile(time: float) -> np.ndarray:
    """Generates an IMU measurement for a circular trajectory,
    where the robot only rotates about the z-axis and the acceleration
    points towards the center of the circle.
    """

    omega_true = np.array([0, 0, -1.0]).reshape((-1, 1))
    accel_true = np.array([0, -3.0, 9.81]).reshape((-1, 1))
    u: np.ndarray = np.vstack([omega_true, accel_true])

    return u


@dataclass
class InitializationParameters:
    """Class to store filter initialization parameters for the filter."""

    perfect_initialization: bool = False
    sigma_phi_init: float = 0.1
    sigma_v_init: float = 0.1
    sigma_r_init: float = 0.1
    sigma_bg_init: float = 0.01
    sigma_ba_init: float = 0.01

    def __post_init__(self):
        """Set all sigmas to zero if we want perfect initialization"""
        if self.perfect_initialization:
            self.sigma_phi_init = 0
            self.sigma_v_init = 0
            self.sigma_r_init = 0
            self.sigma_bg_init = 0
            self.sigma_ba_init = 0


def generate_bias_trajectories(
    sim_config: SimulationConfig, n_meas: int, dt: float
):
    # Generate bias trajectories
    sigma_gyro_bias_dt = sim_config.sigma_gyro_bias_ct / np.sqrt(dt)
    sigma_accel_bias_dt = sim_config.sigma_accel_bias_ct / np.sqrt(dt)

    # Accel bias
    accel_bias = np.zeros((3, n_meas))
    accel_bias[:, [0]] = sim_config.init_accel_bias
    for i in range(1, n_meas):
        accel_bias[:, [i]] = accel_bias[
            :, [i - 1]
        ] + dt * sigma_accel_bias_dt * np.random.randn(3, 1)

    # Gyro Bias
    gyro_bias = np.zeros((3, n_meas))
    gyro_bias[:, [0]] = sim_config.init_gyro_bias
    for i in range(1, n_meas):
        gyro_bias[:, [i]] = gyro_bias[
            :, [i - 1]
        ] + dt * sigma_gyro_bias_dt * np.random.randn(3, 1)

    return gyro_bias, accel_bias


def generate_initialization(
    init_true_state: IMUState,
    delta_xi: np.ndarray,
    init_params: InitializationParameters,
) -> Dict:

    init_estimate = init_true_state.copy()
    init_estimate.plus(delta_xi)

    if init_params.perfect_initialization:
        init_covariance = np.eye(15) * 1e-12
    else:
        init_covariance = np.eye(15)
        init_covariance[0:3, 0:3] *= init_params.sigma_phi_init**2
        init_covariance[3:6, 3:6] *= init_params.sigma_v_init**2
        init_covariance[6:9, 6:9] *= init_params.sigma_r_init**2
        init_covariance[9:12, 9:12] *= init_params.sigma_bg_init**2
        init_covariance[12:15, 12:15] *= init_params.sigma_ba_init**2

    init = {
        "state": init_estimate,
        "covariance": init_covariance,
    }

    return init


def generate_random_delta_x(
    init_params: InitializationParameters = InitializationParameters(),
):
    """Generates a delta_x based on some initialization parameters.

    Parameters
    ----------
    init_params : InitializationParameters
    """
    delta_x = np.vstack(
        (
            np.random.randn(3, 1) * init_params.sigma_phi_init,
            np.random.randn(3, 1) * init_params.sigma_v_init,
            np.random.randn(3, 1) * init_params.sigma_r_init,
            np.random.randn(3, 1) * init_params.sigma_bg_init,
            np.random.randn(3, 1) * init_params.sigma_ba_init,
        )
    )

    return delta_x


def generate_inertial_nav_example(sim_config: SimulationConfig):
    """Data generation for the inertial navigation example.

    The simulated situation here is a vehicle moving in circles along the plane,
    where the true acceleration and angular velocity resolved in the body frame
    is constant.

    The data generator class is only used to generate the groundtruth navigation states,
    (orientation, position, velocity), and noiseless measurements.
    Bias and noise are added to all measurements afterwards.
    """

    # Create IMU process model
    imu_process_model = IMUKinematics(None)

    # Generate landmarks positions and measurement models
    landmarks = generate_landmark_positions(
        sim_config.cylinder_radius,
        sim_config.max_height,
        sim_config.n_level,
        sim_config.n_landmarks_per_level,
    )
    meas_cov = np.identity(3) * sim_config.sigma_landmark_sensor**2
    meas_model_list = [
        PointRelativePosition(pos, meas_cov) for pos in landmarks
    ]

    # Create data generator and generate noiseless navigation states
    data_gen = DataGenerator(
        imu_process_model,
        input_func=input_profile,
        input_covariance=None,
        input_freq=sim_config.input_freq,
        meas_model_list=meas_model_list,
        meas_freq_list=sim_config.landmark_sensor_freq,
    )

    # Initial state
    nav_state_0 = SE23.from_components(
        np.identity(3),
        np.array([3, 0, 0]).reshape((-1, 1)),
        np.array([0, 3, 0]).reshape((-1, 1)),
    )
    x0 = IMUState(
        nav_state_0,
        np.zeros(3),
        np.zeros(3),
        stamp=sim_config.t_start,
        state_id=0,
    )

    states_true, input_list, meas_list = data_gen.generate(
        x0,
        sim_config.t_start,
        sim_config.t_end,
        noise=False,
    )

    # TODO: Figure out a good way to generate bias trajectories with
    # datagen.py
    # Generate bias trajectories, modelled as random walk processes
    bg, ba = generate_bias_trajectories(
        sim_config, len(states_true), 1 / sim_config.input_freq
    )

    # Assign biases to state
    for i, state in enumerate(states_true):
        state.bias_gyro = bg[:, [i]]
        state.bias_accel = ba[:, [i]]

    # Add bias and noise to inputs
    bias = np.vstack([bg, ba])
    Q_dt = np.eye(6)
    dt = 1.0 / sim_config.input_freq
    Q_dt[0:3, 0:3] *= sim_config.sigma_gyro_ct**2 / dt
    Q_dt[3:6, 3:6] *= sim_config.sigma_accel_ct**2 / dt

    for i, u in enumerate(input_list):
        current_bias = bias[:, i].reshape((-1, 1))
        current_noise = randvec(Q_dt)
        u.value = u.value + current_bias + current_noise

    # Add noise to the measurements
    for meas in meas_list:
        meas.value = meas.value + randvec(meas.model.covariance(None))

    sim_data = {}
    sim_data["states_true"] = states_true
    sim_data["input_list"] = input_list
    sim_data["meas_list"] = meas_list
    sim_data["landmarks"] = landmarks

    return sim_data
