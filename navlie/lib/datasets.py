"""A collection of simulated dataset examples."""

from navlie.datagen import DataGenerator
from typing import List
import navlie as nav
from navlie.lib import (
    IMUKinematics,
    IMU,
    IMUState,
    SE3State,
    RangePoseToAnchor,
    BodyFrameVelocity,
    GlobalPosition,
)
import numpy as np
from pylie import SE3, SE23


class SimulatedPoseRangingDataset(nav.Dataset):
    """
    Simulated dataset of a simple rigid robot moving in 3D space with range
    measurements to anchors. The robot has two ranging tags mounted on the
    robot, and measures its own velocity resolved in its body frame.
    """

    def __init__(
        self,
        x0: SE3State = None,
        Q: np.ndarray = None,
        R: float = 0.01,
        input_freq: int = 100,
        meas_freq: int = 10,
        t_start: int = 0,
        t_end: int = 30,
        noise_active: bool = True,
    ):
        """

        Parameters
        ----------
        x0 : SE3State, optional
            Initial state, by default set to identity.
        Q : np.ndarray with shape (6,6), optional
            Covariance associated with angular velocity and translational
            velocity measurements (in that order), by default None
        R : float, optional
            covariance of the range measurements, by default 0.01
        input_freq : int, optional
            frequency of the velocity measurements, by default 100
        meas_freq : int, optional
            frequency of the range measurements, by default 10
        t_start : int, optional
            start time of the simulation, by default 0
        t_end : int, optional
            end time of the simulation, by default 30
        noise_active : bool, optional
            If true, generated measurements will be noisy, by default True
        """
        if x0 is None:
            x0 = SE3State(np.identity(4), stamp=t_start, direction="right")
        if Q is None:
            Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])

        process_model = BodyFrameVelocity(Q)

        def input_profile(t, x):
            return np.array(
                [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
            )

        range_models = [
            RangePoseToAnchor([1, 0, 0], [0.17, 0.17, 0], R),
            RangePoseToAnchor([1, 0, 0], [-0.17, 0.17, 0], R),
            RangePoseToAnchor([-1, 0, 0], [0.17, 0.17, 0], R),
            RangePoseToAnchor([-1, 0, 0], [-0.17, 0.17, 0], R),
            RangePoseToAnchor([0, 2, 0], [0.17, 0.17, 0], R),
            RangePoseToAnchor([0, 2, 0], [-0.17, 0.17, 0], R),
            RangePoseToAnchor([0, 2, 2], [0.17, 0.17, 0], R),
            RangePoseToAnchor([0, 2, 2], [-0.17, 0.17, 0], R),
        ]

        dg = DataGenerator(
            process_model,
            input_profile,
            Q,
            input_freq,
            range_models,
            meas_freq,
        )
        gt_data, input_data, meas_data = dg.generate(
            x0, t_start, t_end, noise=noise_active
        )

        self.gt_data = gt_data
        self.input_data = input_data
        self.meas_data = meas_data

        # Also store the process model used to generate the simulation
        self.process_model = process_model
        self.noise_active = noise_active

    def get_ground_truth(self) -> List[SE3State]:
        return self.gt_data

    def get_input_data(self) -> List[nav.StampedValue]:
        return self.input_data

    def get_meas_data(self) -> List[nav.Measurement]:
        return self.meas_data


class SimulatedInertialGPSDataset(nav.Dataset):
    def __init__(
        self,
        x0: IMUState = None,
        Q: np.ndarray = None,
        R: np.ndarray = 0.01,
        input_freq: int = 200,
        meas_freq: int = 5,
        t_start: int = 0,
        t_end: int = 50,
        noise_active: bool = True,
    ):
        if x0 is None:
            init_nav_state = SE23.from_components(
                np.identity(3),
                np.array([0, 3, 3]).reshape((-1, 1)),
                np.array([3, 0, 0]).reshape((-1, 1)),
            )
            init_gyro_bias = np.array([0.02, 0.03, -0.04]).reshape((-1, 1))
            init_accel_bias = np.array([0.01, 0.02, 0.05]).reshape((-1, 1))
            x0 = IMUState(
                init_nav_state,
                init_gyro_bias,
                init_accel_bias,
                stamp=t_start,
                state_id=0,
                direction="right",
            )

        if Q is None:
            imu_freq = 200
            Q_c = np.eye(12)
            Q_c[0:3, 0:3] *= 0.01**2  # Gyro continuous-time covariance
            Q_c[3:6, 3:6] *= 0.01**2  # Accel continuous-time covariance
            Q_c[6:9, 6:9] *= (
                0.0001**2
            )  # Gyro random-walk continuous-time covariance
            Q_c[9:12, 9:12] *= (
                0.0001**2
            )  # Accel random-walk continuous-time covariance
            dt = 1 / imu_freq
            Q = Q_c / dt

        if isinstance(R, float):
            R = R * np.identity(3)

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
            omega = np.array([0.1, 0, 0.5]).reshape((-1, 1)) + bias_gyro
            a_a = np.array(
                [-3 * np.cos(stamp), -3 * np.sin(stamp), -9 * np.sin(3 * stamp)]
            ).reshape((-1, 1))
            accel = C_ab.T @ a_a + bias_accel - C_ab.T @ g_a

            # Generate a random input to drive the bias random walk
            Q_bias = Q[6:, 6:]
            bias_noise = nav.randvec(Q_bias)

            u = IMU(omega, accel, stamp, bias_noise[0:3], bias_noise[3:6])
            return u

        # Create process and measurement models and generate data
        process_model = IMUKinematics(Q)
        meas_model_list = [GlobalPosition(R)]
        data_gen = DataGenerator(
            process_model,
            input_func=input_profile,
            input_covariance=Q,
            input_freq=input_freq,
            meas_model_list=meas_model_list,
            meas_freq_list=meas_freq,
        )
        # Generate all data
        gt_data, input_data, meas_data = data_gen.generate(
            x0, t_start, t_end, noise=noise_active
        )

        # Zero-out the random walk values (thus creating "noise")
        if noise_active:
            for u in input_data:
                u.bias_gyro_walk = np.array([0, 0, 0])
                u.bias_accel_walk = np.array([0, 0, 0])
        # Save all data and process model
        self.gt_data = gt_data
        self.input_data = input_data
        self.meas_data = meas_data
        self.process_model = process_model

    def get_ground_truth(self) -> List[IMUState]:
        return self.gt_data

    def get_input_data(self) -> List[IMU]:
        return self.input_data

    def get_meas_data(self) -> List[nav.Measurement]:
        return self.meas_data
