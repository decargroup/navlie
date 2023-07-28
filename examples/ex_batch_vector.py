"""
This is an example script showing how to run a batch estimator on custom 
process and measurement models using the utilities in navlie.batch. 

The example is a single integrator process model with three range measurement
models.
"""

import navlie as nav
from navlie.lib import SingleIntegrator, RangePointToAnchor, VectorState
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def main():
    # #############################################################################
    # Create the batch estimator with desired settings
    estimator = nav.BatchEstimator(solver_type="GN", max_iters=5)

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

    dg = nav.DataGenerator(
        process_model,
        input_profile,
        input_covariance,
        input_freq,
        range_models,
        range_freqs,
    )

    gt_data, input_data, meas_data = dg.generate(x0, 0, 10, noise=noise_active)

    # ##############################################################################
    # Run batch
    if noise_active:
        x0 = x0.plus(nav.randvec(P0))

    estimate_list, opt_results = estimator.solve(
        x0, P0, input_data, meas_data, process_model, return_opt_results=True
    )

    print(opt_results["summary"])
    # # The estimate list returns the estimates at each of the
    # # interoceptive and measurement timestamps.
    # Find matching timestamps
    estimate_stamps = [float(x.state.stamp) for x in estimate_list]
    gt_stamps = [x.stamp for x in gt_data]

    matches = nav.associate_stamps(estimate_stamps, gt_stamps)

    est_list = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data[match[1]])
        est_list.append(estimate_list[match[0]])

    # Postprocess the results and plot
    results = nav.GaussianResultList.from_estimates(est_list, gt_list)
    return results

if __name__ == "__main__":
    results = main()
    fig, ax = nav.plot_error(results)
    ax[0].set_title("Position")
    ax[1].set_title("Velocity")
    ax[0].set_xlabel("Time (s)")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
