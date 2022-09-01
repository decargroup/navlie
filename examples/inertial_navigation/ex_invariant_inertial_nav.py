"""A biased inertial navigation example, with 
relative position measurements from known landmarks."""

from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import rospy
from pylie import SE23
from pynav.filters import ExtendedKalmanFilter, run_filter
from pynav.lib.models import (
    IMUKinematics,
    InvariantMeasurement,
    InvariantRelativeLandmark,
)
from pynav.lib.states import IMUState
from pynav.types import Measurement
from pynav.utils import GaussianResult, GaussianResultList, plot_error

import inertial_nav_sim_utils as sim_utils
from rviz_utils.types import PointCloudViz

# This flag can be set to true to visualize the results in RViz.
# Note: this requires the rviz_utils package here :
#   https://bitbucket.org/decargroup/rviz_utils/src/main/
use_rviz = False


def main():
    # Generate the sim data and groundtruth trajetory
    sim_config = sim_utils.SimulationConfig(t_start=0, t_end=30, input_freq=100.0)
    sim_data = sim_utils.generate_inertial_nav_example(sim_config)

    states_gt: List[IMUState] = sim_data["states_true"]

    # Define the process model and create the EKF
    Q_c = np.eye(12)
    Q_c[0:3, 0:3] *= sim_config.sigma_gyro_ct**2
    Q_c[3:6, 3:6] *= sim_config.sigma_accel_ct**2
    Q_c[6:9, 6:9] *= sim_config.sigma_gyro_bias_ct**2
    Q_c[9:12, 9:12] *= sim_config.sigma_accel_bias_ct**2
    process_model = IMUKinematics(Q_c)
    ekf = ExtendedKalmanFilter(process_model=process_model)

    # Define the initial state estimate
    init_params = sim_utils.InitializationParameters()
    delta_xi_init = sim_utils.generate_random_delta_x(init_params)
    true_init_state = states_gt[0]
    true_init_state.direction = "left"
    init = sim_utils.generate_initialization(
        states_gt[0],
        delta_xi_init,
        init_params,
    )

    # Create invariant measurements to be used in the filter
    meas_list: List[Measurement] = sim_data["meas_list"]
    inv_meas_list = []
    for meas in meas_list:
        invariant_meas_model = InvariantRelativeLandmark(meas.value, meas.model)
        invariant_meas = InvariantMeasurement(meas, "right", invariant_meas_model)
        inv_meas_list.append(invariant_meas)

    estimate_list = run_filter(
        ekf,
        init["state"],
        init["covariance"],
        sim_data["input_list"],
        inv_meas_list,
    )

    # Postprocess the results and plot
    results = GaussianResultList(
        [
            GaussianResult(estimate_list[i], states_gt[i])
            for i in range(len(estimate_list))
        ]
    )
    plot_error(results)
    plt.show()

    state_list = [estimate.state for estimate in estimate_list]
    if use_rviz:
        visualize(state_list, sim_data["landmarks"])


def visualize(
    states: List[IMUState],
    landmarks: List[np.ndarray],
):
    """Publishes states and landmarks over RViz for visualization.

    Parameters
    ----------
    states : List[IMUStates]
        IMUStates to publish
    landmarks : List[np.ndarray]
        Landmarks resolved in the inertial frame to publish
    """
    # Imports from rviz_utils package
    from rviz_utils.types import OdometryViz, PathViz
    from rviz_utils.visualization import Visualization

    # Plot states gt
    viz = Visualization()

    imu_state = OdometryViz(pub_name="imu_state")
    imu_path = PathViz(pub_name="imu_path")
    landmarks_viz = PointCloudViz(pub_name="landmarks")

    viz.add_element("imu_state", imu_state)
    viz.add_element("imu_path", imu_path)
    viz.add_element("landmarks", landmarks_viz)

    for i, state in enumerate(states[:-1]):

        viz.update_element("imu_state", state.attitude, state.position)
        viz.update_element("imu_path", state.attitude, state.position)
        viz.update_element("landmarks", None, landmarks)

        next_time = states[i + 1].stamp
        dt = next_time - state.stamp

        rospy.sleep(rospy.Duration(dt, 0))


if __name__ == "__main__":
    main()
