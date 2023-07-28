from navlie.lib.models import SingleIntegrator, RangePointToAnchor, VectorState
from navlie import run_filter, randvec, GaussianResultList, SigmaPointKalmanFilter, DataGenerator, plot_error
import numpy as np
from typing import List
import time

"""
This is an example script showing how to define a custom process model and
measurement model, generate data using those models, and then run an Sigma Point 
Kalman Filter on that data.
"""
def main():
    # ##############################################################################
    # Problem Setup

    x0 = VectorState(np.array([1, 0]), stamp=0)
    P0 = np.diag([1, 1])
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_models = [
        RangePointToAnchor([0, 4], R),
        RangePointToAnchor([-2, 0], R),
        RangePointToAnchor([2, 0], R),
    ]
    range_freqs = [50, 50, 50]
    process_model = SingleIntegrator(Q)
    input_profile = lambda t, x: np.array([np.sin(t), np.cos(t)])
    input_covariance = Q
    input_freq = 180
    noise_active = True
    # ##############################################################################
    # Data Generation

    dg = DataGenerator(
        process_model,
        input_profile,
        input_covariance,
        input_freq,
        range_models,
        range_freqs,
    )

    gt_data, input_data, meas_data = dg.generate(x0, 0, 10, noise=noise_active)

    # ##############################################################################
    # Run Filter
    if noise_active:
        x0 = x0.plus(randvec(P0))
        

    ukf = SigmaPointKalmanFilter(process_model, method= 'cubature', iterate_mean=False)
    # ukf = UnscentedKalmanFilter(process_model, iterate_mean=False) # Equivalent syntax!

    start_time = time.time()
    estimates = run_filter(ukf, x0, P0, input_data, meas_data)
    print("Average filter computation frequency (Hz):")
    print(1 / ((time.time() - start_time) / len(input_data)))
    results = GaussianResultList.from_estimates(estimates, gt_data)
    return results


if __name__ == "__main__":
    results = main()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(results.value[:, 0], results.value[:, 1], label="Estimate")
    ax.plot(
        results.value_true[:, 0], results.value_true[:, 1], label="Ground truth"
    )
    ax.set_title("Trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()

    fig, axs = plot_error(results)
    fig.suptitle("Estimation Error")
    plt.show()
