from typing import List

import matplotlib.pyplot as plt
import numpy as np

from navlie.lib.datasets import SimulatedInertialGPSDataset
from navlie.filters import ExtendedKalmanFilter, run_filter
from navlie.utils import  GaussianResultList, plot_error, randvec

np.set_printoptions(precision=3, suppress=True, linewidth=200)
np.random.seed(0)

# ##############################################################################
# Create simulated IMU/GPS data
data = SimulatedInertialGPSDataset()
gt_states = data.get_ground_truth()
input_data = data.get_input_data()
meas_data = data.get_meas_data()

# Filter initialization
P0 = np.eye(15)
P0[0:3, 0:3] *= 0.1**2
P0[3:6, 3:6] *= 0.1**2
P0[6:9, 6:9] *= 0.1**2
P0[9:12, 9:12] *= 0.01**2
P0[12:15, 12:15] *= 0.01**2
x0 = gt_states[0].plus(randvec(P0))
# ##############################################################################
# Run filter
ekf = ExtendedKalmanFilter(data.process_model)
estimate_list = run_filter(ekf, x0, P0, input_data, meas_data)

# Postprocess the results and plot
results = GaussianResultList.from_estimates(estimate_list, gt_states)

# ##############################################################################
# Plot results
from navlie.utils import plot_poses

fig = plt.figure()
ax = plt.axes(projection="3d")
states_list = [x.state for x in estimate_list]
plot_poses(states_list, ax, line_color="tab:blue", step=20, label="Estimate")
plot_poses(gt_states, ax, line_color="tab:red", step=500, label="Groundtruth")
ax.legend()

fig, axs = plot_error(results)
axs[0, 0].set_title("Attitude")
axs[0, 1].set_title("Velocity")
axs[0, 2].set_title("Position")
axs[0, 3].set_title("Gyro bias")
axs[0, 4].set_title("Accel bias")
axs[-1, 2]

plt.show()
