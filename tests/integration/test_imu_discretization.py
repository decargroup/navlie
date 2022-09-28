from pylie import SE23, SO3
from pynav.lib.imu import IMU, IMUKinematics
from pynav.lib.states import SE23State
import numpy as np


class ClassicIMU:
    """
    The classical IMU equations shown in Forster et al. (2017).
    """
    def __init__(self):
        self._g = np.array([0, 0, -9.80665])

    def evaluate(self, x: SE23State, u: IMU, dt: float) -> SE23State:
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
    accel = [3 * np.sin(t), 0, 9.80665+0.1]
    gyro = [0, 0, 3]
    return IMU(gyro, accel, t)


def do_dead_reckoning(duration=10):
    """ 
    This function performs dead reckoning in three different ways:

    1. The ground truth. Using the classical IMU equations, but with an
        ultra-fine time step (10000 Hz) to eliminate discretization errors.
    2. The classical IMU equations with a 100 Hz time step. This is the benchmark
        and considered the standard in literature.
    3. The new method using the compact SE_2(3) formulation.

    The expected result is that methods 1 and 3 produce the same result,
    while method 2 has discretization errors.
    """
    t_imu = np.arange(0, duration, 0.05)  # 100 Hz
    imu_list = [input_profile(t) for t in t_imu]

    dt_gt = 1 / 10000

    x0 = SE23State(SE23.identity(), 0, direction="left")
    x_gt = x0.copy()

    model1 = ClassicIMU()
    model2 = IMUKinematics(np.identity(6))

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


    # New model
    x_n = x0.copy()
    x_n_list = []
    for k in range(1, len(imu_list)):
        dt = imu_list[k].stamp - imu_list[k - 1].stamp
        x_n = model2.evaluate(x_n, imu_list[k - 1], dt)
        x_n_list.append(x_n.copy())


    # Return ground truth, classic, new.
    return x_gt_list, x_c_list, x_n_list

def test_imu_discretization():
    x_gt_list, _, x_n_list = do_dead_reckoning(duration=3)

    r_gt = np.array([x.position for x in x_gt_list])
    r_n = np.array([x.position for x in x_n_list])
    assert np.allclose(r_gt, r_n, atol=1e-3)

if __name__ == "__main__":
    # test_imu_discretization()   

    # Visualize the results by running this test as a script.
    x_gt_list, x_c_list, x_n_list = do_dead_reckoning()
    r_gt = np.array([x.position for x in x_gt_list])
    v_gt = np.array([x.velocity for x in x_gt_list])

    r_c = np.array([x.position for x in x_c_list])
    v_c = np.array([x.velocity for x in x_c_list])

    r_n = np.array([x.position for x in x_n_list])
    v_n = np.array([x.velocity for x in x_n_list])

    import matplotlib.pyplot as plt
    import matplotlib

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2], label="Ground truth")
    ax.plot(r_c[:, 0], r_c[:, 1], r_c[:, 2], label="Classic")
    ax.plot(r_n[:, 0], r_n[:, 1], r_n[:, 2], label="New")
    ax.legend()
    e_c = np.linalg.norm(r_c - r_gt, axis=1)
    e_n = np.linalg.norm(r_n - r_gt, axis=1)
    t_imu = [x.stamp for x in x_gt_list]
    matplotlib.style.use("seaborn")
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(t_imu, e_c, label="Classic")
    ax.plot(t_imu, e_n, label="New")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Norm of position error [m]")
    plt.show()