import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
import seaborn as sns
from navlie.lib.camera import PinholeCamera
from pymlg import SO3, SE3

sns.set_style("whitegrid")

def test_plot_poses_3d():
    N = 500
    x0 = nav.lib.SE3State([0.3, 3, 4, 0, 0, 0], direction="right")
    process_model = nav.lib.BodyFrameVelocity(np.zeros(6))

    dt = 0.1
    T = 20
    stamps = np.arange(0, T, dt)

    x_traj = [x0.copy()]
    u = nav.lib.VectorInput([0.1, 0.3, 0, 1, 0, 0])
    x = x0.copy()
    for _ in stamps:
        x = process_model.evaluate(x, u, dt)
        x_traj.append(x.copy())

    # Plot the trajectory in 3D
    fig, ax = nav.plot_poses(x_traj)

    # Plot SE(3) poses in 2D
    fig, ax2 = nav.plot_poses(x_traj, plot_2d=True)

def test_plot_poses_2d():
    x0 = nav.lib.SE2State([0.3, 3, 0], direction="right")
    process_model = nav.lib.BodyFrameVelocity(np.zeros(3))
    
    dt = 0.1
    T = 50
    stamps = np.arange(0, T, dt)

    x_traj = [x0.copy()]
    u = nav.lib.VectorInput([0.1, 0.3, 0])
    x = x0.copy()
    for _ in stamps:
        x = process_model.evaluate(x, u, dt)
        x_traj.append(x.copy())

    # Test plotting SE(2) poses
    fig, ax = nav.plot_poses(x_traj)

def test_plot_camera_poses():
    """Tests the camera pose visualization function."""
    C_bc = PinholeCamera.get_cam_to_enu()
    poses = []

    for i in range(5):
        C_ab = SO3.Exp([0.0, 0.0, i / 2.0])
        C_ac = C_ab @ C_bc
        pose = nav.lib.SE3State(
            value=SE3.from_components(C_ac, np.array([i / 2, i / 4, 0.0]))
        )
        poses.append(pose)

    fig, ax = nav.utils.plot_camera_poses(poses, scale=0.15, line_thickness=1.0, color="tab:blue")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

if __name__ == "__main__":
    test_plot_poses_3d()
    test_plot_poses_2d()
    test_plot_camera_poses()
    plt.show()
