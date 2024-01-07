import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == "__main__":
    test_plot_poses_3d()
    test_plot_poses_2d()
    plt.show()
