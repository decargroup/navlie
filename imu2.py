from pylie import SE23, SO3
from pynav.lib.models import Imu, ImuKinematics
from pynav.lib.states import SE23State
from pynav.types import ProcessModel, StampedValue
import numpy as np
from scipy.linalg import expm
from math import factorial


class ClassicIMU:
    def __init__(self):
        self._g = np.array([0, 0, -9.80665])

    def evaluate(self, x: SE23State, u: Imu, dt: float) -> SE23State:
        C = x.attitude
        v = x.velocity
        r = x.position
        r = r + v * dt + 0.5 * (C @ u.accel + self._g) * dt**2
        v = v + (C @ u.accel + self._g) * dt
        C = C @ SO3.Exp(u.gyro * dt)
        x.attitude = C
        x.position = r
        x.velocity = v
        x.stamp += dt
        return x

def input_profile(t):
    accel = [3 * np.sin(t), 0, 9.80665 + 0.1]
    gyro = [2, 3, 3]
    return Imu(accel, gyro, t)


t_imu = np.arange(0, 10, 0.05)  # 100 Hz
imu_list = [input_profile(t) for t in t_imu]

dt_gt = 1 / 10000

x0 = SE23State(SE23.identity(), 0, direction="left")
x_gt = x0.copy()

model1 = ClassicIMU()
model2 = ImuKinematics()

# GROUND TRUTH
x_gt_list = []
for k in range(1, len(imu_list)):

    dt = imu_list[k].stamp - imu_list[k - 1].stamp

    # We will integrate more finely between IMU measurements to
    # eliminate discretization errors.
    sub_t = np.linspace(
        imu_list[k - 1].stamp, imu_list[k].stamp, int(dt / dt_gt)
    )
    for k_sub in range(1, len(sub_t)):
        dt_sub = sub_t[k_sub] - sub_t[k_sub - 1]
        x_gt = model1.evaluate(x_gt, imu_list[k - 1], dt_sub)
    x_gt_list.append(x_gt.copy())

# Classical model
x_c = x0.copy()
x_c_list = []
for k in range(1, len(imu_list)):
    dt = imu_list[k].stamp - imu_list[k - 1].stamp
    x_c = model1.evaluate(x_c, imu_list[k - 1], dt)

    x_c_list.append(x_c.copy())

r_c = np.array([x.position for x in x_c_list])
v_c = np.array([x.velocity for x in x_c_list])

# New model
x_n = x0.copy()
x_n_list = []
for k in range(1, len(imu_list)):
    dt = imu_list[k].stamp - imu_list[k - 1].stamp
    x_n = model2.evaluate(x_n, imu_list[k - 1], dt)
    jac = model2.jacobian(x_n, imu_list[k - 1], dt)
    x_n_list.append(x_n.copy())

r_n = np.array([x.position for x in x_n_list])
v_n = np.array([x.velocity for x in x_n_list])


import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure()
ax = plt.axes(projection="3d")

r_gt = np.array([x.position for x in x_gt_list])
v_gt = np.array([x.velocity for x in x_gt_list])
ax.plot(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2], label="Ground truth")
ax.plot(r_c[:, 0], r_c[:, 1], r_c[:, 2], label="Classic")
ax.plot(r_n[:, 0], r_n[:, 1], r_n[:, 2], label="New")
ax.legend()
e_c = np.linalg.norm(r_c - r_gt, axis=1)
e_n = np.linalg.norm(r_n - r_gt, axis=1)

matplotlib.style.use("seaborn")
fig = plt.figure()
ax = plt.axes()
ax.plot(t_imu[1:], e_c, label="Classic")
ax.plot(t_imu[1:], e_n, label="New")
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Norm of position error [m]")
plt.show()
