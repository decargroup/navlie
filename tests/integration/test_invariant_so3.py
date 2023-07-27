from navlie.lib.states import SO3State
from navlie.lib.models import (
    BodyFrameVelocity,
    InvariantMeasurement,
    Magnetometer,
    Gravitometer,
)
from navlie.datagen import DataGenerator
from navlie.filters import ExtendedKalmanFilter, run_filter
from navlie.utils import (
    GaussianResultList,
    MonteCarloResult,
    randvec,
)
import numpy as np
import navlie as nav

def generate_so3_results():
    x0 = SO3State([1,2,3], 0.0, direction="left")
    P0 = 0.1 * np.identity(3)
    Q = 0.1**2 * np.identity(3)

    # Define the process model and measurement models.
    process_model = BodyFrameVelocity(Q)
    mag_model = Magnetometer(0.1**2 * np.identity(3))
    grav_model = Gravitometer(0.1**2 * np.identity(3))

    # Generate some data
    dg = DataGenerator(
        process_model,
        lambda t, x: np.array([1, 2, 3]),
        Q,
        100,
        [grav_model, mag_model],
        1,
    )
    state_true, input_list, meas_list = dg.generate(x0, 0, 30, True)


    # Run the regular filter
    x0_check = x0.plus(randvec(P0))
    ekf = ExtendedKalmanFilter(process_model=process_model)
    estimate_list = run_filter(ekf, x0_check, P0, input_list, meas_list)
    results = GaussianResultList.from_estimates(estimate_list, state_true)
    return results


def test_reasonable_nees_so3():
    np.random.seed(0)
    results = generate_so3_results()
    results = MonteCarloResult([results])
    N = 1
    nees_in_correct_region = np.count_nonzero(
        results.average_nees < 2 * results.nees_upper_bound(0.99)
    )
    nt = results.average_nees.shape[0]
    # Proportion of time NEES remains below 2*upper_bound bigger than 95%
    assert nees_in_correct_region / nt > 0.80

    # Make sure we essentially never get a completely absurd NEES.
    nees_in_correct_region = np.count_nonzero(
        results.average_nees < 50 * results.nees_upper_bound(0.99)
    )
    assert nees_in_correct_region / nt > 0.9999


if __name__ == "__main__":
    test_reasonable_nees_so3()