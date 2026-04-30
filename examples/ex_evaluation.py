"""Showcases a basic example of how to evaluate results from
multiple SLAM algorithms on a single dataset.

We load in a groundtruth trajectory and an estimated trajectory from the EuroC MAV
dataset, then generate a second estimated trajectory by adding some position drift
to the first estimate.

For each estimate, APE/RPE metrics are computed. For more detail on these metrics,
see the paper:

Z. Zhang and D. Scaramuzza, "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry",
IROS 2018:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8593941
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from navlie.utils.plot import plot_poses
from navlie.utils.common import load_tum_trajectory
from navlie.utils.alignment import associate_and_align_trajectories
from navlie.utils.common import randvec

from navlie.utils.evaluation import (
    compute_ate,
    compute_rpe_over_segment_lengths,
    plot_rpe_boxplot,
)
from evo.core.metrics import PoseRelation

np.random.seed(0)

cur_dir = os.path.dirname(os.path.realpath(__file__))

sns.set_theme(style="whitegrid")
plt.rc("lines", linewidth=1.5)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")
plt.rc("grid", alpha=0.9)

if __name__ == "__main__":
    gt_path = os.path.join(cur_dir, "../data/MH_01_easy.txt")
    est_path = os.path.join(cur_dir, "../data/MH_01_easy_est.txt")

    # Load some groundtruth and some estimated states
    gt_states = load_tum_trajectory(gt_path)
    est_states_1 = load_tum_trajectory(est_path)

    # Simulate a second trajectory with some position drift
    est_states_2 = []
    accumulated_xi = np.zeros(3)
    noise_cov = np.identity(3) * 0.01**2
    for state in est_states_1:
        accumulated_xi += randvec(noise_cov).ravel()
        noisy_state = state.copy()
        noisy_state.position += accumulated_xi.ravel()
        est_states_2.append(noisy_state)

    state_est_list = [est_states_1, est_states_2]

    # Figure to plot the groundtruth and the estimates
    fig, ax = plot_poses(
        gt_states,
        label="groundtruth",
        kwargs_line={"color": "black"},
        step=2000,
    )

    # Loop through each of the state estimates, and compute metrics for each one
    rpe_dict_att = {}
    rpe_dict_pos = {}
    ate_dict_att = {}
    ate_dict_pos = {}
    for i, est_states in enumerate(state_est_list):
        # # Align the trajectories and associate the timestamps
        gt_states, est_states_aligned, _ = associate_and_align_trajectories(
            gt_states,
            est_states,
        )

        # Plot the aligned trajectories
        fig, ax = plot_poses(
            est_states_aligned,
            ax=ax,
            label=f"estimate {i+1} (aligned)",
            kwargs_line={"color": f"C{i}"},
            step=2000,
        )

        ate_att, ate_pos = compute_ate(
            gt_states,
            est_states,
            align=True,
        )
        ate_dict_att[f"estimate_{i+1}"] = ate_att
        ate_dict_pos[f"estimate_{i+1}"] = ate_pos

        # Compute RPE over different segment lengths
        rpe_att = compute_rpe_over_segment_lengths(
            gt_states,
            est_states,
            align=True,
            pose_relation=PoseRelation.rotation_angle_deg,
        )
        rpe_pos = compute_rpe_over_segment_lengths(
            gt_states,
            est_states,
            align=True,
            pose_relation=PoseRelation.translation_part,
        )
        rpe_dict_att[f"estimate_{i+1}"] = rpe_att
        rpe_dict_pos[f"estimate_{i+1}"] = rpe_pos

    # Plot the RPE boxplots for both orientation and position
    fig, ax = plt.subplots(2, 1, sharex=True)
    plot_rpe_boxplot(rpe_dict_att, ax=ax[0])
    plot_rpe_boxplot(rpe_dict_pos, ax=ax[1])
    ax[1].set_xlabel("Segment Length (m)")
    ax[0].set_ylabel("RPE (degrees)")
    ax[1].set_ylabel("RPE (m)")
    ax[0].grid(True)
    ax[1].grid(True)
    fig.suptitle("RPE over Segment Lengths")
    fig.tight_layout()

    # Print a summary table of the ATE results
    ate_df = pd.DataFrame(
        {
            "ATE (deg)": ate_dict_att,
            "ATE (m)": ate_dict_pos,
        }
    )
    ate_df.index.name = "Estimate"
    print("\nATE Summary:")
    print(ate_df.to_string(float_format="{:.3f}".format))
    plt.show()
