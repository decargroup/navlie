from pynav.states import SE3State
from pynav.models import BodyFrameVelocity, RangePoseToAnchor
from pynav.datagen import DataGenerator
from pynav.filters import ExtendedKalmanFilter
from pynav.utils import GaussianResult, GaussianResultList, plot_error 
import time
from pylie import SE3
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme()

# %% Problem setup
x0 = SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0)
P0 = 1 * np.identity(6)
Q =np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
process_model = BodyFrameVelocity(Q)

def input_profile(t):
    return np.array([np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0])
       


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

# %% Data generation
dg = DataGenerator(process_model, input_profile, Q, 200, range_models, 10)
state_gt, input_data, meas_data = dg.generate(x0, 0, 10, noise=True)

# %% Run Filter
ekf = ExtendedKalmanFilter(x0, P0, process_model)

meas_idx = 0
start_time = time.time()
y = meas_data[meas_idx]
results: List[GaussianResult] = []
for k in range(len(input_data) - 1):
    u = input_data[k]

    # Fuse any measurements that have occurred.
    while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

        ekf.correct(y)
        meas_idx += 1
        if meas_idx < len(meas_data):
            y = meas_data[meas_idx]

    ekf.predict(u)
    results.append(GaussianResult(ekf.x, ekf.P, state_gt[k]))


print("Average filter computation frequency (Hz):")
print(1 / ((time.time() - start_time) / len(input_data)))

r = GaussianResultList(results)

# %%

sns.set_theme()
fig, axs = plot_error(r)
axs[-1][0].set_xlabel("Time (s)")
axs[-1][1].set_xlabel("Time (s)")
axs[0][0].set_title("Rotation Error")
axs[0][1].set_title("Translation Error")
plt.show()

# %%
