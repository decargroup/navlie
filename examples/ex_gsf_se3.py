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

    x0_check = x0.plus(nav.randvec(P0))

    estimate_list = nav.gsf.run_gsf_filter(
        gsf, x0_check, P0, input_list, meas_list
    )

    results = [
        nav.imm.IMMResult(estimate_list[i], state_true[i])
        for i in range(len(estimate_list))
    ]

    return nav.imm.IMMResultList(results)

N = 1
results = nav.monte_carlo(gsf_trial, N)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(1, 1)
    ax.plot(results.stamp, results.average_nees, label="IMM NEES")
    ax.plot(
        results.stamp, results.expected_nees, color="r", label="Expected NEES"
    )
    ax.plot(
        results.stamp,
        results.nees_lower_bound(0.99),
        color="k",
        linestyle="--",
        label="99 percent c.i.",
    )
    ax.plot(
        results.stamp,
        results.nees_upper_bound(0.99),
        color="k",
        linestyle="--",
    )
    ax.set_title("{0}-trial average NEES".format(results.num_trials))
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.legend()

    plt.show()