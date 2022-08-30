"""A biased inertial navigation example, with absolute position measurements"""

import math
from pynav.lib.states import IMUState
from pynav.lib.models import (
    IMUKinematics,
    InvariantMeasurement,
    Magnetometer,
    Gravitometer,
)
import rospy
from pynav.datagen import DataGenerator
from pynav.filters import ExtendedKalmanFilter, run_filter
from pynav.types import StampedValue
from pynav.utils import GaussianResult, GaussianResultList, plot_error
from pylie import SE23
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Visualization
from rviz_utils.types import OdometryViz, PathViz
from rviz_utils.visualization import Visualization

# Start and end times
t_start = 0
t_end = 30

# Specify continuous-time noise values
sigma_gyro_ct = 0.001
sigma_accel_ct = 0.001
sigma_gyro_bias_ct = 1e-5
sigma_accel_bias_ct = 1e-5

# Discritize noise
input_freq = 100.0
dt = 1 / input_freq

Q_c = np.eye(12)
Q_c[0:3, 0:3] *= sigma_gyro_ct**2
Q_c[3:6, 3:6] *= sigma_accel_ct**2
Q_c[6:9, 6:9] *= sigma_gyro_bias_ct**2
Q_c[9:12, 9:12] *= sigma_accel_bias_ct**2

Q_dt = scipy.linalg.block_diag(
    np.identity(3) * sigma_gyro_ct / math.sqrt(dt),
    np.identity(3) * sigma_accel_ct / math.sqrt(dt),
    np.identity(3) * sigma_gyro_bias_ct / math.sqrt(dt),
    np.identity(3) * sigma_accel_bias_ct / math.sqrt(dt),
)

nav_state_0 = SE23.from_components(
    np.identity(3),
    np.array([3, 0, 0]).reshape((-1, 1)),
    np.array([0, 3, 0]).reshape((-1, 1)),
)

x0 = IMUState(
    nav_state_0,
    np.zeros(3),
    np.zeros(3),
    stamp=t_start,
    state_id=0,
)

# Data generation
# The simulated situation here is a vehicle moving in circles along the plane,
# where the true acceleration and angular velocity resolved in the body frame
# is constant.
# TODO: see if this can also be done with datagen.py?
omega_true = np.array([0, 0, -1.0]).reshape((-1, 1))
accel_true = np.array([0, -3.0, 9.81]).reshape((-1, 1))
u: np.ndarray = np.vstack([omega_true, accel_true])

times = np.arange(t_start, t_end, 1 / input_freq)


# Create IMU process model
imu_process_model = IMUKinematics(Q_c)

states_gt = []
x = x0.copy()
x.stamp = times[0]
states_gt = [x.copy()]
for i in range(0, len(times) - 1):
    t_k = times[i]
    u_k = StampedValue(u.copy(), t_k)
    x = imu_process_model.evaluate(x, u_k, dt)
    states_gt.append(x.copy())

# Plot states gt
viz = Visualization()

imu_state = OdometryViz(pub_name="imu_state")
imu_path = PathViz(pub_name="imu_path")

viz.add_element("imu_state", imu_state)
viz.add_element("imu_path", imu_path)
for i, state in enumerate(states_gt[:-1]):

    viz.update_element("imu_state", state.attitude, state.position)
    viz.update_element("imu_path", state.attitude, state.position)

    next_time = states_gt[i + 1].stamp
    dt = next_time - state.stamp

    rospy.sleep(rospy.Duration(0.01, 0))


position = np.array([state.position for state in states_gt]).T
velocity = np.array([state.velocity for state in states_gt])

fig, ax = plt.subplots(1, 1)
ax.plot(times, position)
plt.show()

print(position.shape[0])
