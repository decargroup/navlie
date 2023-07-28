import navlie as nav
import time
import numpy as np
np.random.seed(0)

def main():
    # ##############################################################################
    # Create simulated pose ranging data
    data = nav.lib.SimulatedPoseRangingDataset()
    gt_states = data.get_ground_truth()
    input_data = data.get_input_data()
    meas_data = data.get_meas_data()

    # %% ###########################################################################
    # Perturb initial groundtruth state to initialize filter
    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1, 1, 1])
    x0 = gt_states[0].plus(nav.randvec(P0))
    x = nav.StateWithCovariance(x0, P0)

    # Run Filter - try an EKF or an IterEKF
    # ekf = ExtendedKalmanFilter(process_model)
    ekf = nav.IteratedKalmanFilter(data.process_model)

    meas_idx = 0
    start_time = time.time()
    y = meas_data[meas_idx]
    results_list = []
    for k in range(len(input_data) - 1):
        results_list.append(nav.GaussianResult(x, gt_states[k]))

        u = input_data[k]
        
        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

            x = ekf.correct(x, y, u)
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        dt = input_data[k + 1].stamp - x.stamp
        x = ekf.predict(x, u, dt)
        


    print("Average filter computation frequency (Hz):")
    print(1 / ((time.time() - start_time) / len(input_data)))

    results = nav.GaussianResultList(results_list)
    return results

if __name__ == "__main__":
    results = main()
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme()
    fig, axs = nav.plot_error(results)
    axs[-1][0].set_xlabel("Time (s)")
    axs[-1][1].set_xlabel("Time (s)")
    axs[0][0].set_title("Rotation Error")
    axs[0][1].set_title("Translation Error")
    plt.show()
