import typing
import numpy as np
from navlie.lib.datasets import SimulatedPoseRangingDataset
from navlie.lib.states import SE3State
from navlie.utils import plot_poses

from pymlg import SO3, SE3
import matplotlib.pyplot as plt

from navlie.utils.alignment import associate_and_align_trajectories, state_list_to_evo_traj, evo_traj_to_state_list

def test_conversion_to_evo():
    """Converts a list of SE3State to an evo trajectory and back."""
    dataset = SimulatedPoseRangingDataset()
    poses_1 = dataset.get_ground_truth()
    evo_traj = state_list_to_evo_traj(poses_1)
    poses_2 = evo_traj_to_state_list(evo_traj)

    for pose_1, pose_2 in zip(poses_1, poses_2):
        assert np.allclose(pose_1.position, pose_2.position, atol=1e-5)
        assert np.allclose(pose_1.attitude, pose_2.attitude, atol=1e-5)
        assert np.allclose(pose_1.stamp, pose_2.stamp, atol=1e-5)

def test_associate_and_align_trajectories():
    """Test utilizing evo to associate and align two trajectories."""
    # Load in the sample EuRoC dataset
    # Create an initial trajectory
    dataset = SimulatedPoseRangingDataset()
    poses_1 = dataset.get_ground_truth()

    # Create a second trajectory that is a similarity transformation of the first
    scale = 0.9
    rot = SO3.Exp(np.array([0.01, 0.01, 1.2]))
    pos = np.array([1.0, 2.0, 0.5])

    poses_2: typing.List[SE3State] = []
    for pose in poses_1:
        new_position = scale * rot @ pose.position + pos
        new_att = rot @ pose.attitude
        new_pose = SE3State(
            value=SE3.from_components(new_att, new_position),
            stamp=pose.stamp,
            direction=pose.direction,
        )
        poses_2.append(new_pose)

    fig, ax = plot_poses(
        poses_1, line_color="tab:blue", step=None, label="Original Trajectory"
    )
    fig, ax = plot_poses(
        poses_2, ax=ax, line_color="tab:red", step=None, label="Transformed Trajectory"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    # Try aligning the first trajectory to the second
    traj_ref, aligned_traj_sim3, transformation_info = associate_and_align_trajectories(
        poses_2,
        poses_1,
        correct_scale=True,
        align=True,
        verbose=True
    )

    # Check that the computed transformation matches the true one
    assert np.allclose(transformation_info["scale"], scale, atol=1e-5)
    assert np.allclose(transformation_info["rotation"], rot, atol=1e-5)
    assert np.allclose(transformation_info["position"], pos, atol=1e-5)

    # Verify visually
    fig, ax = plot_poses(
        aligned_traj_sim3,
        ax=ax,
        line_color="tab:green",
        step=None,
        label="Aligned Trajectory (Sim3)",
    )   

    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_conversion_to_evo()
    test_associate_and_align_trajectories()