from navlie.lib import SE2State, BodyFrameVelocity, RangePoseToAnchor

import navlie as nav
import numpy as np
from typing import List


"""
This example runs a Gaussian Sum filter to estimate the state 
that is on a Lie group. The performance is compared to an EKF that is 
initialized at a wrong state.
"""

def main():

    # Create the process model noise profile
    Q = np.diag([0.1**2, 0.1, 0.1])
    process_model = BodyFrameVelocity(Q)


    # Measurement model
    R = 0.1**2
    range_models = [
        RangePoseToAnchor([-5, 5],[0, 0], R),
        RangePoseToAnchor([ 5, 5],[0, 0], R),
    ]

    # Setup
    x0 = SE2State([0, -5, 0], stamp=0.0) # theta, x, y
    P0 = np.diag([0.1**2, 0.1, 0.1])

    def input_profile(t, u):
        return np.array( [0.0, 0.1, np.cos(t)])

    input_freq = 10
    dt = 1 / input_freq
    t_max = dt * 100
    measurement_freq = 5

    # gsf filter
    gsf = nav.GaussianSumFilter(process_model)

    dg = nav.DataGenerator(
        process_model,
        input_profile,
        Q,
        input_freq,
        range_models,
        measurement_freq,
    )


    def gsf_trial(trial_number: int) -> List[nav.GaussianResult]:
        """
        A single Gaussian Sum Filter trial
        """
        np.random.seed(trial_number)
        state_true, input_list, meas_list = dg.generate(x0, 0, t_max, True)
        
        # Initial state estimates
        x = [SE2State([0, -5, 0], stamp=0.0),
            SE2State([0,  5, 0], stamp=0.0)]
        x = [x_.plus(nav.randvec(P0)) for x_ in x]
        
        x0_check = nav.lib.MixtureState(
            [nav.StateWithCovariance(_x, P0) for _x in x], [1/len(x) for _ in x]
        )

        estimate_list = nav.run_gsf_filter(
            gsf, x0_check, input_list, meas_list
        )

        results = [
            nav.MixtureResult(estimate_list[i], state_true[i])
            for i in range(len(estimate_list))
        ]

        return nav.MixtureResultList(results)

    def ekf_trial(trial_number: int) -> List[nav.GaussianResult]:
        """
        A single trial in a monte carlo experiment. This function accepts the trial
        number and must return a list of GaussianResult objects.
        """

        # By using the trial number as the seed for the random generator, we can
        # make sure our experiments are perfectly repeatable, yet still have
        # independent noise samples from trial-to-trial.
        np.random.seed(trial_number)

        state_true, input_list, meas_list = dg.generate(x0, 0, t_max, noise=True)
        x0_check = SE2State([0, 5, 0], stamp=0.0)
        x0_check = x0_check.plus(nav.randvec(P0))
        ekf = nav.ExtendedKalmanFilter(BodyFrameVelocity(Q))

        estimate_list = nav.run_filter(ekf, x0_check, P0, input_list, meas_list)
        return nav.GaussianResultList.from_estimates(estimate_list, state_true)
    
    N = 1 # Trial number
    return gsf_trial(N), ekf_trial(N)

if __name__ == "__main__":
    results_gsf, results_ekf = main()


    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = nav.plot_error(results_gsf, label = 'gsf')
    nav.plot_error(results_ekf, axs=ax, label = 'ekf')
    ax[0].set_title("Error plots")
    ax[0].set_ylabel("Error (rad)")
    ax[1].set_ylabel("Error (m)")
    ax[2].set_ylabel("Error (m)")
    ax[2].set_xlabel("Time (s)")
    plt.legend()
    plt.show()