import os
import numpy as np
import typing
from tqdm import tqdm

from pymlg import SE3
from navlie.lib.states import SE3State
from navlie.bspline import SE3Bspline
from navlie.lib.models import BodyFrameVelocity, VectorInput

import matplotlib.pyplot as plt

cur_dir = os.path.dirname(os.path.realpath(__file__))


def test_bspline():
    """Tests the B-Spline class by fitting a B-Spline to a 
    constant velocity trajectory and ensuring that the velocities computed 
    from the B-Spline match the true velocities .
    """

    # Constant velocity input
    vel_input = VectorInput(
        value=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        stamp=0.0,
    )

    # Propagate an initial pose forward with the constant velocity input
    dt = 0.01
    init_pose = SE3State(
        value=np.eye(4),
        stamp=0.0,
    )
    gt_poses = [init_pose.copy()]
    process_model = BodyFrameVelocity(Q=np.identity(6))
    for i in range(1000):
        next_pose = process_model.evaluate(gt_poses[-1], vel_input, dt)
        next_pose.stamp += dt
        gt_poses.append(next_pose)

    # Fit a B-Spline to the trajectory
    bspline = SE3Bspline(gt_poses, verbose=True)

    # Query the B-Spline to get the poses and velocities at a set of times
    query_stamps = [x.stamp for x in gt_poses]
    for stamp in query_stamps:
        pose_spline = bspline.get_pose(stamp)
        omega_b_ba, vel_a_spline = bspline.get_velocity(stamp)
        alpha_spline, accel_a_spline = bspline.get_acceleration(stamp)

        if pose_spline is not None and vel_a_spline is not None:
            vel_b = pose_spline.attitude.T @ vel_a_spline
            # Check that the spline velocities match the inputs
            assert np.allclose(omega_b_ba.ravel(), vel_input.value[:3])
            assert np.allclose(vel_b.ravel(), vel_input.value[3:6])
            # Check angular acceleration is zero
            assert np.allclose(alpha_spline.ravel(), np.zeros(3)) 


if __name__ == "__main__":
    test_bspline()
