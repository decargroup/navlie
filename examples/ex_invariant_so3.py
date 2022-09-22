from pynav.lib.states import SO3State
from pynav.lib.models import (
    BodyFrameVelocity,
    InvariantMeasurement,
    Magnetometer,
    Gravitometer,
)
from pynav.datagen import DataGenerator
from pynav.filters import ExtendedKalmanFilter, run_filter
from pynav.utils import GaussianResult, GaussianResultList, plot_error
from pylie import SO3
import numpy as np
import matplotlib.pyplot as plt

# ##############################################################################
# Problem Setup

# Define the initial state
x0 = SO3State(SO3.random(), 0.0, direction="left")
P0 = 1 * np.identity(3)
Q = 0.1**2 * np.identity(3)
noise_active = False 

# Define the process model and measurement models.
process_model = BodyFrameVelocity(Q)
mag_model = Magnetometer(0.1**2 * np.identity(3))
grav_model = Gravitometer(0.1**2 * np.identity(3))


# ##############################################################################
# Data generation

dg = DataGenerator(
    process_model,
    lambda t, x: np.array([1, 2, 3]),
    Q,
    100,
    [mag_model, grav_model],
    1,
)
state_true, input_list, meas_list = dg.generate(x0, 0, 30, noise_active)

# ##############################################################################
# Run the regular filter
x0.direction = "right"
ekf = ExtendedKalmanFilter(process_model=process_model)
estimate_list = run_filter(ekf, x0, P0, input_list, meas_list)
results = GaussianResultList(
    [
        GaussianResult(estimate_list[i], state_true[i])
        for i in range(len(estimate_list))
    ]
)

plot_error(results)

# ##############################################################################
# Run the invariant filter

# **************** Conversion to Invariant Measurements ! *********************
invariants = [InvariantMeasurement(meas, "right") for meas in meas_list]
# *****************************************************************************

x0.direction = "left"
ekf = ExtendedKalmanFilter(process_model=process_model)
estimate_list = run_filter(ekf, x0, P0, input_list, invariants)

results_invariant = GaussianResultList(
    [
        GaussianResult(estimate_list[i], state_true[i])
        for i in range(len(estimate_list))
    ]
)

plot_error(results_invariant)
plt.show()
