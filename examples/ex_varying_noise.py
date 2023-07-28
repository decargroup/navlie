# %%

from navlie.lib import VectorState, DoubleIntegrator, RangePointToAnchor
from navlie import (
    run_filter,
    monte_carlo,
    GaussianResult,
    GaussianResultList,
    plot_error,
    plot_nees,
    randvec,
    DataGenerator,
    ExtendedKalmanFilter


)
import numpy as np
from typing import List
"""
This is an example script showing how to define time-varying noise matrices
and to then run an EKF using these same noise matrices. 
"""

def main():
    x0_true = VectorState(np.array([1, 0]), stamp=0.0)
    P0 = np.diag([0.5, 0.5])
    R = 0.01**2
    Q = 0.1 * np.identity(1)
    N = 10 # Number MC trials

    range_models = [
        RangePointToAnchor([0.0], R),
    ]


    t_max = 10
    range_freqs = [10]
    input_freq = 10
    dt = 1/input_freq

    def Q_profile(t):
        if t <= t_max/4:
            Q = 1
        if t > t_max/4 and t <= t_max*3/4:
            Q = 100
        if t > t_max*3/4:
            Q = 1
        Q = np.array(Q).reshape((1,1))
        return Q

    class VaryingNoiseDoubleIntegrator(DoubleIntegrator):
        def __init__(self, Q_profile):
            self.Q_profile = Q_profile
            super().__init__(Q_profile(0))

        def covariance(self, x, u, dt) -> np.ndarray:
            self._Q = self.Q_profile(x.stamp)
            return super().covariance(x, u, dt)

    # For data generation, the Q for the process model does not matter as
    # only the evaluate method is used. 
    process_model = VaryingNoiseDoubleIntegrator(Q_profile)
    input_profile = lambda t, x: np.sin(t)

    dg = DataGenerator(
        process_model,
        input_profile,
        Q_profile,
        input_freq,
        range_models,
        range_freqs
    )


    ekf = ExtendedKalmanFilter(process_model)

    def ekf_trial(trial_number:int) -> List[GaussianResult]:
        """
        A single trial in a monte carlo experiment. This function accepts the trial
        number and must return a list of GaussianResult objects.
        """

        np.random.seed(trial_number)
        state_true, input_data, meas_data = dg.generate(x0_true, 0, t_max, noise=True)

        x0_check = x0_true.plus(randvec(P0))
        
        estimates = run_filter(ekf, x0_check, P0, input_data, meas_data, True)

        return GaussianResultList.from_estimates(estimates, state_true)

    # %% Run the monte carlo experiment
    results = monte_carlo(ekf_trial, num_trials=N)

    return results
# %% Plot
if __name__ == "__main__":
    results = main()
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_theme()
    sns.set_style("whitegrid")

    fig, ax = plot_nees(results)

    if results.num_trials < 15:

        fig, axs = plt.subplots(2, 1)
        axs: List[plt.Axes] = axs
        for result in results.trial_results:
            plot_error(result, axs = axs)

        axs[0].set_title("Estimation error")
        axs[1].set_xlabel("Time (s)")

    plt.show()