"""
This is an example script showing how to run a batch estimator on custom 
process and measurement models.
"""

from navlie.lib import (
    BodyFrameVelocity,
    RangePoseToAnchor,
    SE3State
)
import navlie as nav
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ##########################################################################
    # Create the batch estimator with desired settings
    estimator = nav.BatchEstimator(solver_type="GN", max_iters=20)

    # ##########################################################################
    # Problem Setup
    t_end = 4
    x0 = SE3State([0, 0, 0, 0, 0, 0], stamp=0.0)
    P0 = 0.1**2 * np.identity(6)
    R = 0.1**2
    Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
    range_models = [
        RangePoseToAnchor([1, 0, 0], [0.17, 0.17, 0], R),
        RangePoseToAnchor([1, 0, 0], [-0.17, 0.17, 0], R),
        RangePoseToAnchor([-1, 0, 0], [0.17, 0.17, 0], R),
        RangePoseToAnchor([-1, 0, 0], [-0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 0], [0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 0], [-0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 2], [0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 2], [-0.17, 0.17, 0], R),
    ]

    range_freqs = 20
    process_model = BodyFrameVelocity(Q)
    input_profile = lambda t, x: np.array(
        [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
    )

    input_freq = 100
    noise_active = True


    # Generate data with no noise
    dg = nav.DataGenerator(
        process_model,
        input_profile,
        Q,
        input_freq,
        range_models,
        range_freqs,
    )

    state_true, input_list, meas_list = dg.generate(x0, 0, t_end, noise_active)

    # Run batch
    estimate_list, opt_results = estimator.solve(
        x0,
        P0,
        input_list,
        meas_list,
        process_model,
        return_opt_results=True,
    )

    print(opt_results["summary"])


    results = nav.GaussianResultList.from_estimates(estimate_list, state_true)
    return results


if __name__ == "__main__":
    results = main()
    fig, ax = nav.plot_error(results)
    ax[-1][0].set_xlabel("Time (s)")
    ax[-1][1].set_xlabel("Time (s)")
    ax[0][0].set_title("Orientation Error")
    ax[0][1].set_title("Position Error")
    plt.show()
