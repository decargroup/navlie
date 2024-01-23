"""A simple stereo bundle adjustment example using navlie's nonlinear least
squares solver."""

import typing

import matplotlib.pyplot as plt
import numpy as np
from pymlg import SE3

from navlie.batch.problem import Problem
from navlie.batch.residuals import Residual
from navlie.composite import CompositeState
from navlie.lib.camera import PinholeCamera, PoseMatrix
from navlie.lib.models import CameraProjection
from navlie.lib.states import SE3State, VectorState
from navlie.types import Measurement, State
from navlie.utils import plot_poses, randvec

def main():
    # Simulation parameters
    n_landmarks_width = 10
    n_landmarks_height = 10
    depth_landmark = 1.0
    # Generate poses and landmarks for the problem
    poses = generate_poses()
    landmarks = generate_landmarks(
        n_landmarks_width, n_landmarks_height, depth_landmark
    )
    print(f"Number of poses generated: {len(poses)}")
    print(f"Number of landmarks generated: {len(landmarks)}")

    # Create a camera model and generate measurements
    # T_bc is the transformation between the robot frame and the camera frame, w
    # where the camera frame has the z-axis pointing through the optical center.
    T_bc_0 = PoseMatrix(
        SE3.from_components(PinholeCamera.get_cam_to_enu(), np.array([0.0, 0.0, 0.0]))
    )
    T_bc_1 = PoseMatrix(
        SE3.from_components(PinholeCamera.get_cam_to_enu(), np.array([0.0, -0.05, 0.0]))
    )
    camera_0 = PinholeCamera(385, 385, 325, 235, 640, 480, 1.0, T_bc_0)
    camera_1 = PinholeCamera(385, 385, 325, 235, 640, 480, 1.0, T_bc_1)
    cameras = [camera_0, camera_1]
    measurements = generate_measurements(poses, landmarks, cameras)

    # Generate initial guess for optimizer by perturbing all states from groundtruth
    poses_init: typing.Dict[str, SE3State] = {}
    for i, (pose_id, pose) in enumerate(poses.items()):
        # Initialize first pose to groundtruth
        if i == 0:
            pose_value = pose.copy()
        else:
            pose_value = pose @ SE3.Exp(randvec(np.identity(6) * 0.0001))
        poses_init[pose_id] = SE3State(
            value=pose_value, state_id=pose_id, direction="right"
        )

    landmarks_init: typing.Dict[str, VectorState] = {}
    for landmark_id, landmark in landmarks.items():
        landmark_value = landmark + randvec(np.identity(3) * 0.01).ravel()
        landmarks_init[landmark_id] = VectorState(
            value=landmark_value, state_id=landmark_id
        )

    # Create and solve the problem
    problem = Problem(solver="GN")

    # Add poses and landamrks to problem
    for pose_id, pose in poses_init.items():
        problem.add_variable(pose_id, pose)
    for landmark_id, landmark in landmarks_init.items():
        problem.add_variable(landmark_id, landmark)

    # Hold the first pose constant
    problem.set_variables_constant(["p0"])

    # Add factors to problem
    for meas in measurements:
        keys = meas.state_id
        residual = ReprojectionResidual(keys, meas)
        problem.add_residual(residual)

    # Solve the problem
    opt_results = problem.solve()
    variables_opt = opt_results["variables"]
    print(opt_results["summary"])
    # Extract estimates
    poses_est: typing.Dict[str, SE3State] = {}
    landmarks_est: typing.Dict[str, VectorState] = {}
    for pose_id in poses_init.keys():
        poses_est[pose_id] = variables_opt[pose_id]
    for landmark_id in landmarks_init.keys():
        landmarks_est[landmark_id] = variables_opt[landmark_id]

    # Plot initial and estimated poses
    init_pose_list = list(poses_init.values())
    init_landmarks_list = [x.value for x in landmarks_init.values()]
    fig, ax = plot_bundle_adjustment(init_landmarks_list, init_pose_list)
    fig.suptitle("Initial Guess")
    opt_pose_list  = list(poses_est.values())
    opt_landmarks_list = [x.value for x in landmarks_est.values()]
    fig, ax = plot_bundle_adjustment(opt_landmarks_list, opt_pose_list)
    fig.suptitle("Optimized Solution")

class ReprojectionResidual(Residual):
    def __init__(
        self,
        keys: typing.List[typing.Hashable],
        meas: Measurement,
    ):
        super().__init__(keys)
        self.meas = meas
        # Evaluate the square root information a single time since it does not
        # depend on the state in this case
        self.sqrt_information = self.meas.model.sqrt_information([])

    def evaluate(
        self,
        states: typing.List[State],
        compute_jacobians: typing.List[bool] = None,
    ) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
        # Create a composite state to evaluate the measurement model
        eval_state = CompositeState(states)
        # Evaluate the measurement model
        y_check = self.meas.model.evaluate(eval_state)
        # Compute the residual
        error = self.meas.value - y_check

        L = self.sqrt_information
        error = L.T @ error

        if compute_jacobians:
            jacobians = [None] * len(states)
            full_jac = -self.meas.model.jacobian(eval_state)
            if compute_jacobians[0]:
                jacobians[0] = L.T @ full_jac[:, :6]
            if compute_jacobians[1]:
                jacobians[1] = L.T @ full_jac[:, 6:]

            return error, jacobians
        return error


def generate_poses() -> typing.Dict[str, np.ndarray]:
    """Generates some poses for the BA problem."""
    poses: typing.Dict[np.ndarray] = {}
    poses["p0"] = SE3.Exp(np.array([0.2, 0, -0.2, 0.0, 0.4, 0.2]))
    poses["p1"] = SE3.Exp(np.array([0, 0.1, 0, 0.1, 0.2, 0.1]))
    poses["p2"] = SE3.Exp(np.array([0.5, 0.5, 0.2, 0, 0.1, 0.3]))
    return poses


def generate_landmarks(
    num_landmarks_width: int,
    num_landmarks_height: int,
    depth_landmark: float,
) -> typing.Dict[str, np.ndarray]:
    """Generates a grid of landmarks in the plane at a fixed depth."""
    widths = np.linspace(0, 0.3, num_landmarks_width)
    heights = np.linspace(0, 0.3, num_landmarks_height)

    landmarks: typing.Dict[str, np.ndarray] = {}
    landmark_num = 0
    for width in widths:
        for height in heights:
            landmark_id = "l" + str(landmark_num)
            landmarks[landmark_id] = np.array([depth_landmark, width, height])
            landmark_num += 1
    return landmarks


def generate_measurements(
    poses: typing.Dict[str, np.ndarray],
    landmarks: typing.Dict[str, np.ndarray],
    cameras: typing.List[PinholeCamera],
) -> typing.List[Measurement]:
    """Generates measurements of landmarks taken from a number of camera
    poses."""
    meas_list: typing.List[Measurement] = []
    for pose_id, pose in poses.items():
        for landmark_id, landmark in landmarks.items():
            for camera in cameras:
                # Evaluate noiseless measurement
                # Create a PoseMatrix object from the pose matrix
                noiseless_y = camera.evaluate(PoseMatrix(pose), landmark)
                # Add noise
                cov_noise = np.identity(2) * camera.sigma**2
                y_noise = noiseless_y + randvec(cov_noise).ravel()
                # Check if measurement is valid
                if not camera.is_measurement_valid(y_noise):
                    continue
                state_ids = [pose_id, landmark_id]
                meas_model = CameraProjection(pose_id, landmark_id, camera)
                meas = Measurement(y_noise, state_id=state_ids, model=meas_model)
                meas_list.append(meas)

    return meas_list


def plot_bundle_adjustment(
    landmarks: typing.List[np.ndarray],
    pose_list: typing.List[SE3State],
    ax: plt.Axes = None,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    landmarks = np.array(landmarks) 
    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        landmarks[:, 2],
        c="tab:blue",
    )
    plot_poses(
        pose_list,
        ax=ax,
        arrow_length=0.1,
        step=1,
        line_color="tab:blue",
        label="Initial",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return fig, ax

if __name__ == "__main__":
    main()
    plt.show()
