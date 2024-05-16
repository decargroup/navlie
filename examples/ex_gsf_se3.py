from navlie.lib import SE2State, BodyFrameVelocity, RangePoseToAnchor

import navlie as nav
import numpy as np
from typing import List


"""
This example runs an Interacting Multiple Model filter to estimate the process model noise matrix
for a state that is on a Lie group. The performance is compared to an EKF that knows the ground
truth process model noise. 
"""

# Create the process model noise profile
Q = np.diag([0.1**2, 0.1, 0.1])
process_model = BodyFrameVelocity(Q)


# Measurement model
R = 0.1**2
range_models = [
    RangePoseToAnchor([-5,5],[0, 0], R),
    RangePoseToAnchor([ 5,5],[0, 0], R),
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
gsf = nav.gsf.GaussianSumFilter(process_model)

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
    A single Interacting Multiple Model Filter trial
    """
    np.random.seed(trial_number)
    state_true, input_list, meas_list = dg.generate(x0, 0, t_max, True)
    
    # Initial state estimates
    x = [SE2State([0, -5, 0], stamp=0.0),
         SE2State([0,  5, 0], stamp=0.0)]
    x = [x_.plus(nav.randvec(P0)) for x_ in x]
    
    weights = [1, 1]
    x0_check = nav.gsf.GMMState(
        [nav.StateWithCovariance(_x, P0) for _x in x], weights
    )

    estimate_list = nav.gsf.run_gsf_filter(
        gsf, x0_check, P0, input_list, meas_list
    )

    results = [
        nav.gsf.GSFResult(estimate_list[i], state_true[i])
        for i in range(len(estimate_list))
    ]

    return nav.gsf.GSFResultList(results)

N = 2
results = gsf_trial(0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = nav.plot_error(results)
    ax[0].set_title("Error plots")
    ax[0].set_ylabel("Error (rad)")
    ax[1].set_ylabel("Error (m)")
    ax[2].set_ylabel("Error (m)")
    ax[2].set_xlabel("Time (s)")
    plt.show()