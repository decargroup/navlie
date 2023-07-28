# %%
"""
This script shows how to use the monte carlo utils. The monte_carlo() 
function will automatically execute, and aggregate the results from a user-
provided callable trial function. Average NEES, its probability bounds, and 
expected NEES are all automatically calculated for you.
"""

from navlie.lib import SE3State, BodyFrameVelocity, RangePoseToAnchor
from navlie import (
    DataGenerator,
    ExtendedKalmanFilter,
    GaussianResult,
    GaussianResultList,
    monte_carlo,
    plot_error,
    plot_nees,
    randvec,
    run_filter
)
import numpy as np
from typing import List


def main():

    x0_true = SE3State([0, 0, 0, 0, 0, 0], stamp=0.0)
    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 0.3**3, 0.3**2, 0.3**2])
    Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
    process_model = BodyFrameVelocity(Q)


    def input_profile(t, x):
        return np.array(
            [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
        )


    range_models = [
        RangePoseToAnchor([1, 0, 0], [0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([1, 0, 0], [-0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([-1, 0, 0], [0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([-1, 0, 0], [-0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([0, 2, 0], [0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([0, 2, 0], [-0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([0, 2, 2], [0.17, 0.17, 0], 0.1**2),
        RangePoseToAnchor([0, 2, 2], [-0.17, 0.17, 0], 0.1**2),
    ]
    dg = DataGenerator(process_model, input_profile, Q, 100, range_models, 10)

    ekf = ExtendedKalmanFilter(process_model)

    def ekf_trial(trial_number: int) -> List[GaussianResult]:
        """
        A single trial in a monte carlo experiment. This function accepts the trial
        number and must return a list of GaussianResult objects.
        """

        # By using the trial number as the seed for the random generator, we can
        # make sure our experiments are perfectly repeatable, yet still have
        # independent noise samples from trial-to-trial.
        np.random.seed(trial_number)

        state_true, input_data, meas_data = dg.generate(x0_true, 0, 10, noise=True)
        x0_check = x0_true.plus(randvec(P0))
        estimates = run_filter(ekf, x0_check, P0, input_data, meas_data, True)
        return GaussianResultList.from_estimates(estimates, state_true)

    #  Run the monte carlo experiment
    N = 20
    results = monte_carlo(ekf_trial, num_trials=N, num_jobs=4)
    return results

if __name__ == "__main__":
    results = main()

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plot_nees(results)

    if results.num_trials < 15:

        fig, axs = plt.subplots(3, 2)
        axs: List[plt.Axes] = axs
        for result in results.trial_results:
            plot_error(result, axs=axs)

        fig.suptitle("Estimation error")
        
    plt.tight_layout()
    plt.show()
