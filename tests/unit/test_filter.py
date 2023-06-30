import pynav as nav
import numpy as np

def test_iterated_ekf():
    """
    Check if we can run the filter with the default options.
    """

    x = nav.lib.VectorState([1, 0], stamp=0.0)
    P = np.diag([1, 1])
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_model =  nav.lib.RangePointToAnchor([0, 4], R)
    process_model = nav.lib.SingleIntegrator(Q)
    y = nav.generate_measurement(x, range_model)
    u = nav.StampedValue([1,2], stamp=0.0)
    
    x = nav.StateWithCovariance(x, P)
    kf = nav.IteratedKalmanFilter(process_model)
    x = kf.correct(x, y, u)
    x = kf.predict(x, u, dt=0.1)


def test_iterated_ekf_no_line_search():
    """
    Check if we can run the filter without line search
    """

    x = nav.lib.VectorState([1, 0], stamp=0.0)
    P = np.diag([1, 1])
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_model =  nav.lib.RangePointToAnchor([0, 4], R)
    process_model = nav.lib.SingleIntegrator(Q)
    y = nav.generate_measurement(x, range_model)
    u = nav.StampedValue([1,2], stamp=0.0)
    
    x = nav.StateWithCovariance(x, P)
    kf = nav.IteratedKalmanFilter(process_model, line_search=False)
    x = kf.correct(x, y, u)
    x = kf.predict(x, u, dt=0.1)

def test_iterated_ekf_equivalence():
    """
    With a single iteration, the iterated EKF should be equivalent to the
    standard EKF. Note that the covariance will be different, because the
    iterated EKF calculates the covariance using the posterior jacobians as
    opposed to the prior jacobians.
    """
    x = nav.lib.VectorState([1, 0], stamp=0.0)
    P = np.diag([1, 1])
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_model =  nav.lib.RangePointToAnchor([0, 4], R)
    process_model = nav.lib.SingleIntegrator(Q)
    y = nav.generate_measurement(x, range_model)
    u = nav.StampedValue([1,2], stamp=0.0)
    
    x = nav.StateWithCovariance(x, P)
    kf = nav.IteratedKalmanFilter(process_model, max_iters=1, line_search=False)
    ekf = nav.ExtendedKalmanFilter(process_model)
    x1 = kf.correct(x, y, u)
    x1 = kf.predict(x1, u, dt=0.1)
    x2 = ekf.correct(x, y, u)
    x2 = ekf.predict(x2, u, dt=0.1)
    assert np.allclose(x1.state.value, x2.state.value)