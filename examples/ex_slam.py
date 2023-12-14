"""A toy SLAM example where we are interested in estimating robot poses and 
3D landmark positions from IMU measurements and relative position measurements 
to the landmarks.

Here, the purpose is simply to show how the default EKF provided in navlie
can be used in SLAM-type problem settings. The structure of the SLAM problem
is not exploited by doing this. For a more efficient EKF implementation for
SLAM, see the document: 
    Simulataneous localization and mapping with the extended Kalman filter by Joan
    Solà (2014).
"""

import typing
import numpy as np
import navlie as nav
from navlie.lib.imu import IMUState
from navlie.lib.datasets import SimulatedInertialLandmarkDataset
from navlie.lib.states import VectorState, CompositeState
from navlie.lib.models import PointRelativePositionSLAM, CompositeInput, CompositeProcessModel
from scipy.linalg import block_diag

class LandmarkProcessModel(nav.ProcessModel):
    def evaluate(self, x: VectorState, t: float, u: np.ndarray):
        return x.copy()

    def jacobian(self, x: VectorState, t: float, u: np.ndarray):
        return np.eye(3)
    
    def covariance(self, x: VectorState, t: float, u: np.ndarray):
        return np.zeros((3, 3))


def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=1000)
    np.random.seed(0)
    # Load simulated dataset with default parameters
    dataset = SimulatedInertialLandmarkDataset(t_start=0, t_end=10.0, R=0.1)
    gt_states = dataset.get_ground_truth()
    input_data = dataset.get_input_data()
    meas_data = dataset.get_measurement_data()
    landmarks = dataset.get_groundtruth_landmarks()

    # Filter initialization - set small covariance on yaw and position
    # as these are unobservable
    P0_imu = np.eye(15)
    P0_imu[0:2, 0:2] *= 0.1**2
    P0_imu[2, 2] *= 1e-15
    P0_imu[3:6, 3:6] *= 0.1**2
    P0_imu[6:9, 6:9] *= 1e-15
    P0_imu[9:12, 9:12] *= 0.01**2
    P0_imu[12:15, 12:15] *= 0.01**2
    init_imu_state = gt_states[0].plus(nav.randvec(P0_imu))

    # Set the state ID to be "r" for robot state and "l" for landmark state
    robot_state_id = "r"
    landmark_state_id = "l"
    init_imu_state.state_id = robot_state_id

    landmark_state_ids: typing.List[str] = []

    # Create a SLAM state that includes both the landmark states and the robot
    # state
    state_list = [init_imu_state]
    P0_landmark = 0.1**2
    P0_landmark_block = np.identity(3 * len(landmarks)) * P0_landmark
    for i, pos in enumerate(landmarks):
        # Perturb the initial landmark position
        perturbed_pos = pos + nav.randvec(P0_landmark * np.eye(3))
        cur_landmark_state_id = landmark_state_id + str(i)
        state_list.append(VectorState(perturbed_pos, 0.0, cur_landmark_state_id))
        landmark_state_ids.append(cur_landmark_state_id)

    init_state = CompositeState(state_list, stamp=init_imu_state.stamp)
    init_cov = block_diag(P0_imu, P0_landmark_block)

    # Convert all measurments to SLAM measurements, where the landmark position
    # is a state to be estimated
    for meas in meas_data:
        current_landmark_id = landmark_state_id + str(meas.model._landmark_id)
        meas.model = PointRelativePositionSLAM(
            robot_state_id, current_landmark_id, meas.model._R
        )

    # Create a composite process model that includes the robot process model and
    # the (constant) landmark process model for each landmark
    landmark_process_model = LandmarkProcessModel()
    process_model_list = [dataset.process_model]
    for i in range(len(landmarks)):
        process_model_list.append(landmark_process_model)
    process_model = CompositeProcessModel(process_model_list)

    composite_inputs = []
    for u in input_data:
        # Create a composite input for each of the landmarks
        input_list = [u]
        for i in range(len(landmarks)):
            input_list.append(None)
        composite_inputs.append(CompositeInput(input_list))

    # ###########################################################################
    # Create and run filter    
    ekf = nav.ExtendedKalmanFilter(process_model)
    estimate_list = nav.run_filter(ekf, init_state, init_cov, composite_inputs, meas_data)

    # Extract the IMU state estimates from the estimate list
    imu_state_list: typing.List[IMUState] = []
    imu_cov_list: typing.List[np.ndarray] = []
    for estimate in estimate_list:
        imu_state = estimate.state.get_state_by_id(robot_state_id)
        imu_state.stamp = estimate.state.stamp
        imu_state_list.append(imu_state)
        imu_state_slice = estimate.state.get_slice_by_id(robot_state_id)
        imu_cov_list.append(estimate.covariance[imu_state_slice, imu_state_slice])

    imu_estimates_list: typing.List[nav.StateWithCovariance] = []
    for state, cov in zip(imu_state_list, imu_cov_list):
        imu_estimates_list.append(nav.StateWithCovariance(state, cov))

    # Extract the final estimated landmark positions
    final_estimate = estimate_list[-1]
    landmark_est_list: typing.List[np.ndarray] = []
    for landmark_id in landmark_state_ids:
        landmark_state = final_estimate.state.get_state_by_id(landmark_id)
        landmark_est_list.append(landmark_state.value)

    # Postprocess the results and plot
    imu_results = nav.GaussianResultList.from_estimates(imu_estimates_list, gt_states)
    return imu_results, landmark_est_list, dataset


if __name__ == "__main__":
    imu_results, landmark_est_list, dataset = main()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    landmarks = np.array(dataset.get_groundtruth_landmarks())
    est_landmarks = np.array(landmark_est_list)
    ax.scatter(est_landmarks[:, 0], est_landmarks[:, 1], est_landmarks[:, 2], marker="x", color="tab:blue")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], color="tab:red")
    nav.plot_poses(imu_results.state, ax, line_color="tab:blue", step=500, label="Estimate")
    nav.plot_poses(
        imu_results.state_true,
        ax,
        line_color="tab:red",
        step=500,
        label="Groundtruth",
    )
    ax.legend()

    sns.set_theme()
    fig, axs = nav.plot_error(imu_results)
    axs[0, 0].set_title("Attitude")
    axs[0, 1].set_title("Velocity")
    axs[0, 2].set_title("Position")
    axs[0, 3].set_title("Gyro bias")
    axs[0, 4].set_title("Accel bias")
    axs[-1, 2]

    plt.show()