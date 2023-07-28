from navlie.lib import (
    BodyFrameVelocity,
    InvariantMeasurement,
    Magnetometer,
    Gravitometer,
    SO3State,
)
import navlie as nav
from pymlg import SO3
import numpy as np


def main():
    # ##########################################################################
    # Problem Setup

    # Define the initial state
    x0 = SO3State(SO3.random(), 0.0, direction="left")
    P0 = 0.5**2 * np.identity(3)
    Q = 0.1**2 * np.identity(3)
    noise_active = True

    # Define the process model and measurement models.
    process_model = BodyFrameVelocity(Q)
    mag_model = Magnetometer(0.1**2 * np.identity(3))
    grav_model = Gravitometer(0.1**2 * np.identity(3))

    # ##########################################################################
    # Data generation

    dg = nav.DataGenerator(
        process_model,
        lambda t, x: np.array([1, 2, 3]),
        Q,
        100,
        [mag_model, grav_model],
        1,
    )
    state_true, input_list, meas_list = dg.generate(x0, 0, 30, noise_active)

    if noise_active:
        x0 = x0.plus(nav.randvec(P0))

    # ##########################################################################
    # Run the regular filter
    ekf = nav.ExtendedKalmanFilter(process_model=process_model)
    estimate_list = nav.run_filter(ekf, x0, P0, input_list, meas_list)
    results_ekf = nav.GaussianResultList.from_estimates(estimate_list, state_true)

    # ##########################################################################
    # Run the invariant filter
    # TODO. Why does this give the exact same thing as the regular EKF?
    # **************** Conversion to Invariant Measurements ! ******************
    invariants = [InvariantMeasurement(meas) for meas in meas_list]
    # **************************************************************************

    ekf = nav.ExtendedKalmanFilter(process_model=process_model)
    estimate_list = nav.run_filter(ekf, x0, P0, input_list, invariants)

    results_invariant = nav.GaussianResultList.from_estimates(estimate_list, state_true)
    return results_ekf, results_invariant


if __name__ == "__main__":
    results_ekf, results_invariant = main()
    import matplotlib.pyplot as plt

    fig, axs = nav.plot_error(results_ekf)
    fig.suptitle("Regular EKF")

    fig, axs = nav.plot_error(results_invariant)
    fig.suptitle("Invariant EKF")
    plt.show()
