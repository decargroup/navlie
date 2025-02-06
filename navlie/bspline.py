"""
A B-Spline on SE(3), useful for generating simulated trajectories  
in 3D space and interoceptive measurements (like IMU measurements)
along the trajectory.

This is essentially a Python implementation of the C++ code found in OpenVINS here:
https://github.com/rpng/open_vins/blob/master/ov_core/src/sim/BsplineSE3.h

The core idea is to utilize cubic B-splines to interpolate between a set of control
point poses. Denoting :math:`\mathbf{T}_{ab} \in SE(3)` as the vehicle pose, the B-Spline is
parameterized by an evenly spaced set of control points :math:`\mathbf{T}_{ab_i}`. The value
of the spline curve at any time :math:`t` is purely a function of four bounding control points, such that

    .. math::
        \mathbf{T}_{ab} (t) = \exp \left( \\tilde{B}_{0, 4} (t) \log \left(\mathbf{T}_{ab_0} \\right) \\right) \prod_{n=1}^3 \exp \left( \\tilde{B}_{i, 4} \\log \left(\mathbf{T}_{ab_{i-1}}^{-1} \mathbf{T}_{ab_i} \\right) \\right)

where :math:`\\tilde{B}_{i, 4}` are the cumulative basis functions.
For detailed derivations for the key equations behind the B-Spline simulator, see
    E. Mueggler, G. Gallego, H. Rebecq, and D. Scaramuzza. Continuous-Time Visual-Inertial Odometry for Event Cameras. 
    IEEE Transactions on Robotics, pages 1–16, 2018.
"""

import numpy as np
import typing
from navlie.lib.states import SE3State
from navlie.utils.common import state_interp
from pymlg import SO3, SE3


class SE3Bspline:
    def __init__(
        self,
        traj_points: typing.List[SE3State],
        verbose: bool = False,
        max_dt: float = 0.1,
    ):
        self.traj_points = traj_points
        # self.verbose = verbose

        # Find the average frequency to use as our uniform points
        dts = [
            traj_points[i].stamp - traj_points[i - 1].stamp
            for i in range(1, len(traj_points))
        ]
        average_dt = np.mean(dts)
        self.dt = average_dt if average_dt > max_dt else max_dt

        # Get the trajectory start and end points
        start_time = traj_points[0].stamp
        self.end_time = traj_points[-1].stamp

        # Create the spline control points
        self.control_points: typing.List[SE3State] = []

        # Interpolate the control points at uniform intervals
        cur_time = start_time
        if verbose:
            print(
                "Generating evenly spaced control points along the trajectory..."
            )

        while True:
            # Get the bounding poses
            idx_poses = self._find_bounding_poses(cur_time, traj_points)

            if idx_poses is None:
                break

            poses = [traj_points[i] for i in idx_poses]
            # Linear interpolate and append to our control points
            cur_control_point = state_interp(cur_time, poses)
            self.control_points.append(cur_control_point)
            cur_time += self.dt

        self.start_time = start_time + 2 * self.dt

        if verbose:
            print(f"Number of control points: {len(traj_points)}")
            print(
                f"Trajectory length in seconds: {(self.end_time - start_time):.3f}"
            )
            print(f"Trajectory start time: {self.start_time:.3f}")
            print(f"Trajectory end time: {self.end_time:.3f}")

    def get_pose(self, stamp: float) -> SE3State:
        """Query the B-Spline to get the pose at a given timestamp."""
        # Get the bounding control points
        control_points = self._find_bounding_control_points(
            stamp, self.control_points
        )
        if control_points is None:
            return None

        pose0, pose1, pose2, pose3 = control_points
        t2 = pose2.stamp
        t1 = pose1.stamp

        # De Boor-Cox matrix scalars
        DT = t2 - t1
        u = (stamp - t1) / DT
        b0, b1, b2 = self._compute_basis_functions(u)

        # Calculate interpolated poses
        A0 = SE3.Exp(b0 * SE3.Log(SE3.inverse(pose0.value) @ pose1.value))
        A1 = SE3.Exp(b1 * SE3.Log(SE3.inverse(pose1.value) @ pose2.value))
        A2 = SE3.Exp(b2 * SE3.Log(SE3.inverse(pose2.value) @ pose3.value))
        pose_interp_np = pose0.value @ A0 @ A1 @ A2
        pose_interp = SE3State(
            value=pose_interp_np,
            stamp=stamp,
        )
        return pose_interp

    def get_velocity(self, stamp) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Gets the angular and translational velocity at a given timestamp.

        The returned quantity is a tuple of numpy arrays, where the first
        quantity is the angular velocity :math:`\mathbf{\omega}_b^{ba}`,
        resolved in the body frame. The second quantity is the translational
        velocity resolved in the inertial frame :math:`\mathbf{v}_a^{ba}`.
        """

        control_points = self._find_bounding_control_points(
            stamp, self.control_points
        )

        if control_points is None:
            return None, None

        # De Boor-Cox matrix scalars
        pose0, pose1, pose2, pose3 = control_points
        t2 = pose2.stamp
        t1 = pose1.stamp

        # De Boor-Cox matrix scalars
        DT = t2 - t1
        u = (stamp - t1) / DT
        b0, b1, b2 = self._compute_basis_functions(u)
        b0dot, b1dot, b2dot = self._compute_basis_functions_dot(u, DT)

        # Calculate interpolated poses
        omega_10 = SE3.Log(SE3.inverse(pose0.value) @ pose1.value)
        omega_21 = SE3.Log(SE3.inverse(pose1.value) @ pose2.value)
        omega_32 = SE3.Log(SE3.inverse(pose2.value) @ pose3.value)
        A0 = SE3.Exp(b0 * omega_10)
        A1 = SE3.Exp(b1 * omega_21)
        A2 = SE3.Exp(b2 * omega_32)
        A0dot = b0dot * SE3.wedge(omega_10) @ A0
        A1dot = b1dot * SE3.wedge(omega_21) @ A1
        A2dot = b2dot * SE3.wedge(omega_32) @ A2

        # Get the interpolated pose and the interpolated velocities
        pose_interp_np = pose0.value @ A0 @ A1 @ A2
        vel_interp_mat = pose0.value @ (
            A0dot @ A1 @ A2 + A0 @ A1dot @ A2 + A0 @ A1 @ A2dot
        )

        # Extract the angular and translational velocities
        omega_b_ba = SO3.vee(pose_interp_np[0:3, 0:3].T @ vel_interp_mat[0:3, 0:3])
        vel_a_ba = vel_interp_mat[0:3, 3].reshape((-1, 1))
        return omega_b_ba, vel_a_ba

    def get_acceleration(self, stamp) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Computes the angular and translational acceleration at a given
        timestamp.
        
        The returned quantity is a tuple of numpy array with dimension [6 x 1],
        where the first entry is the angular acceleration :math:`\alpha_b^{ba}`,
        and the second entry is the translational acceleration resolved in the
        global frame, :math:`\mathbf{a}_a^{ba}`.
        """
        control_points = self._find_bounding_control_points(
            stamp, self.control_points
        )

        if control_points is None:
            return None, None

        # De Boor-Cox matrix scalars
        pose0, pose1, pose2, pose3 = control_points
        t2 = pose2.stamp
        t1 = pose1.stamp

        # De Boor-Cox matrix scalars
        DT = t2 - t1
        u = (stamp - t1) / DT
        b0, b1, b2 = self._compute_basis_functions(u)
        b0dot, b1dot, b2dot = self._compute_basis_functions_dot(u, DT)
        b0ddot, b1ddot, b2ddot = self._compute_basis_functions_ddot(u, DT)

        # Calculate interpolated poses
        omega_10 = SE3.Log(SE3.inverse(pose0.value) @ pose1.value)
        omega_21 = SE3.Log(SE3.inverse(pose1.value) @ pose2.value)
        omega_32 = SE3.Log(SE3.inverse(pose2.value) @ pose3.value)
        omega_10_hat = SE3.wedge(omega_10)
        omega_21_hat = SE3.wedge(omega_21)
        omega_32_hat = SE3.wedge(omega_32)

        A0 = SE3.Exp(b0 * omega_10)
        A1 = SE3.Exp(b1 * omega_21)
        A2 = SE3.Exp(b2 * omega_32)
        A0dot = b0dot * SE3.wedge(omega_10) @ A0
        A1dot = b1dot * SE3.wedge(omega_21) @ A1
        A2dot = b2dot * SE3.wedge(omega_32) @ A2
        A0ddot = b0dot * omega_10_hat @ A0dot + b0ddot * omega_10_hat @ A0
        A1ddot = b1dot * omega_21_hat @ A1dot + b1ddot * omega_21_hat @ A1
        A2ddot = b2dot * omega_32_hat @ A2dot + b2ddot * omega_32_hat @ A2

        # Get the interpolated pose and velocities
        pose_interp_np = pose0.value @ A0 @ A1 @ A2
        vel_interp_mat = pose0.value @ (
            A0dot @ A1 @ A2 + A0 @ A1dot @ A2 + A0 @ A1 @ A2dot
        )
        acc_interp_mat = pose0.value @ (
            A0ddot @ A1 @ A2 + A0 @ A1ddot @ A2 + A0 @ A1 @ A2ddot + 2 * A0dot @ A1dot @ A2
            + 2 * A0 @ A1dot @ A2dot + 2 * A0dot @ A1 @ A2dot
        )
        
        omega_skew = pose_interp_np[0:3, 0:3].T @ vel_interp_mat[0:3, 0:3]
        alpha = SO3.vee(pose_interp_np[0:3, 0:3].T @ (acc_interp_mat[0:3, 0:3] - vel_interp_mat[0:3, 0:3] @ omega_skew))
        a_a_ba = acc_interp_mat[0:3, 3].reshape((-1, 1))
        return alpha, a_a_ba


    def _find_bounding_control_points(
        self,
        stamp: float,
        control_points: typing.List[SE3State],
    ) -> typing.List[SE3State]:
        # Get the two bounding poses for the timestamp
        idx_poses = self._find_bounding_poses(
            stamp,
            control_points,
        )
        if idx_poses is None:
            return None

        if idx_poses[0] == 0:
            return None
        if idx_poses[1] == len(control_points) - 1:
            return None

        # Get the four control points that bound the timestamp
        pose_1 = control_points[idx_poses[0]]
        pose_2 = control_points[idx_poses[1]]
        pose_0 = control_points[idx_poses[0] - 1]
        pose_3 = control_points[idx_poses[1] + 1]
        return [pose_0, pose_1, pose_2, pose_3]

    @staticmethod
    def _find_bounding_poses(
        stamp: float,
        poses: typing.List[SE3State],
    ) -> typing.List[int]:
        """Finds the bounding poses for a given timestamp.

        Returns None if there are no bounding poses, otherwise returns the
        indices in the pose list that bound the timestamp.
        """
        # Check if the timestamp if outside the range of the poses
        if stamp < poses[0].stamp or stamp > poses[-1].stamp:
            return None

        # Find the poses that bound the timestamp
        for i in range(1, len(poses)):
            if poses[i].stamp > stamp:
                return [i - 1, i]

    @staticmethod
    def _compute_basis_functions(u) -> typing.List[float]:
        b0 = (1.0 / 6.0) * (5.0 + 3.0 * u - 3.0 * u * u + u * u * u)
        b1 = (1.0 / 6.0) * (1.0 + 3.0 * u + 3.0 * u * u - 2.0 * u * u * u)
        b2 = (1.0 / 6.0) * (u * u * u)
        return [b0, b1, b2]

    @staticmethod
    def _compute_basis_functions_dot(u: float, dt: float) -> typing.List[float]:
        b0dot = 1.0 / (6.0 * dt) * (3.0 - 6.0 * u + 3.0 * u * u)
        b1dot = 1.0 / (6.0 * dt) * (3.0 + 6.0 * u - 6.0 * u * u)
        b2dot = 1.0 / (6.0 * dt) * (3.0 * u * u)
        return [b0dot, b1dot, b2dot]

    @staticmethod
    def _compute_basis_functions_ddot(
        u: float, dt: float
    ) -> typing.List[float]:
        b0ddot = 1.0 / (6.0 * dt * dt) * (-6.0 + 6.0 * u)
        b1ddot = 1.0 / (6.0 * dt * dt) * (6.0 - 12.0 * u)
        b2ddot = 1.0 / (6.0 * dt * dt) * (6.0 * u)
        return [b0ddot, b1ddot, b2ddot]
