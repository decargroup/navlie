from navlie.lib import SE3State, BodyFrameVelocity, SimulatedPoseRangingDataset
from navlie import SigmaPointKalmanFilter, run_filter, GaussianResultList, plot_error, randvec, UnscentedKalmanFilter
import time
import numpy as np

np.random.seed(0)

def main():
    # ##########################################################################
    # Problem Setup
    x0 = SE3State([0, 0, 0, 0, 0, 0], stamp=0.0, direction="right")
    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1, 1, 1])
    Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
    noise_active = True
    process_model = BodyFrameVelocity(Q)


    data = SimulatedPoseRangingDataset(x0=x0, Q=Q, noise_active=noise_active)
    state_true = data.get_ground_truth()
    input_data = data.get_input_data()
    meas_data = data.get_meas_data()
    if noise_active:
        x0 = x0.plus(randvec(P0))
    # %% #######################################################################
    # Run Filter

    ukf = SigmaPointKalmanFilter(process_model, method = 'unscented', iterate_mean=False)
    # ukf = UnscentedKalmanFilter(process_model, iterate_mean=False) # Equivalent syntax! 

    start_time = time.time()
    estimates = run_filter(ukf, x0, P0, input_data, meas_data)
    print("Average filter computation frequency (Hz):")
    print(1 / ((time.time() - start_time) / len(input_data)))

    results = GaussianResultList.from_estimates(estimates, state_true)
    return results 

if __name__ == "__main__":

    results = main()
    # ##########################################################################
    # Plot
    import matplotlib.pyplot as plt
    fig, axs = plot_error(results)
    axs[-1][0].set_xlabel("Time (s)")
    axs[-1][1].set_xlabel("Time (s)")
    axs[0][0].set_title("Rotation Error")
    axs[0][1].set_title("Translation Error")
    plt.show()
