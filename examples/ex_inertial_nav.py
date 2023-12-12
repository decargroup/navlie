""" 
A slightly more complicated example of a robot localizing itself from relative
position measurements to known landmarks.
"""
import numpy as np
import navlie as nav
from navlie.lib.datasets import SimulatedInertialLandmarkDataset

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    np.random.seed(0)
    # Load simulated dataset with default parameters
    dataset = SimulatedInertialLandmarkDataset(t_start=0, t_end=20)
    gt_states = dataset.get_ground_truth()
    input_data = dataset.get_input_data()
    meas_data = dataset.get_measurement_data()

    # Filter initialization 
    P0 = np.eye(15)
    P0[0:3, 0:3] *= 0.1**2
    P0[3:6, 3:6] *= 0.1**2
    P0[6:9, 6:9] *= 0.1**2
    P0[9:12, 9:12] *= 0.01**2
    P0[12:15, 12:15] *= 0.01**2
    x0 = gt_states[0].plus(nav.randvec(P0))

    # ###########################################################################
    # Run filter
    ekf = nav.ExtendedKalmanFilter(dataset.process_model)
    estimate_list = nav.run_filter(ekf, x0, P0, input_data, meas_data)

    # Postprocess the results and plot
    results = nav.GaussianResultList.from_estimates(estimate_list, gt_states)
    return results, dataset

if __name__ == "__main__":
    results, dataset = main()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    landmarks = np.array(dataset.get_groundtruth_landmarks())
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
    nav.plot_poses(
        results.state, ax, line_color="tab:blue", step=500, label="Estimate"
    )
    nav.plot_poses(
        results.state_true,
        ax,
        line_color="tab:red",
        step=500,
        label="Groundtruth",
    )
    ax.legend()

    sns.set_theme()
    fig, axs = nav.plot_error(results)
    axs[0, 0].set_title("Attitude")
    axs[0, 1].set_title("Velocity")
    axs[0, 2].set_title("Position")
    axs[0, 3].set_title("Gyro bias")
    axs[0, 4].set_title("Accel bias")
    axs[-1, 2]

    plt.show()
