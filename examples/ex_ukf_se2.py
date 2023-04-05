from pynav.lib import SE3State, BodyFrameVelocity
from pynav.filters import SigmaPointKalmanFilter
from pynav.utils import GaussianResult, GaussianResultList, plot_error, randvec
from pynav.types import StateWithCovariance
from pynav.lib.datasets import SimulatedPoseRangingDataset
import time
from pylie import SE3
import numpy as np
from typing import List
np.random.seed(0)

# ##############################################################################
# Problem Setup
x0 = SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0, direction="right")
P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1, 1, 1])
Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
noise_active = True
process_model = BodyFrameVelocity(Q)


data = SimulatedPoseRangingDataset(x0=x0, Q=Q, noise_active=noise_active)
state_true = data.get_ground_truth()
input_data = data.get_input_data()
meas_data = data.get_meas_data()
if noise_active:
    x0 = x0.plus(randvec(P0))
# %% ###########################################################################
# Run Filter
x = StateWithCovariance(x0, P0)

ukf = SigmaPointKalmanFilter(process_model, method = 'cubature', iterate_mean=False)

meas_idx = 0
start_time = time.time()
y = meas_data[meas_idx]
results_list = []
for k in range(len(input_data) - 1):
    results_list.append(GaussianResult(x, state_true[k]))

    u = input_data[k]
    
    # Fuse any measurements that have occurred.
    while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

        x = ukf.correct(x, y, u)
        meas_idx += 1
        if meas_idx < len(meas_data):
            y = meas_data[meas_idx]

    dt = input_data[k + 1].stamp - x.stamp
    x = ukf.predict(x, u, dt)
    


print("Average filter computation frequency (Hz):")
print(1 / ((time.time() - start_time) / len(input_data)))

results = GaussianResultList(results_list)

# ##############################################################################
# Plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
fig, axs = plot_error(results)
axs[-1][0].set_xlabel("Time (s)")
axs[-1][1].set_xlabel("Time (s)")
axs[0][0].set_title("Rotation Error")
axs[0][1].set_title("Translation Error")
plt.show()
