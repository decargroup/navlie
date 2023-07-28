from navlie.lib import VectorState, DoubleIntegrator, RangePointToAnchor
import navlie as nav
import numpy as np


def main():
    # ##############################################################################
    # Problem Setup

    x0 = VectorState(np.array([1, 0, 0, 0]))
    P0 = 0.1**2*np.identity(x0.dof)
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_models = [
        RangePointToAnchor([0, 4], R),
        RangePointToAnchor([-2, 0], R),
        RangePointToAnchor([2, 0], R),
    ]
    range_freqs = [1, 1, 1]
    process_model = DoubleIntegrator(Q)

    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # The TRUE input profile is zero-mean random signal.
    input_profile = lambda t, x: nav.randvec(Q).ravel() # random walk.

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    input_covariance = Q
    input_freq = 100

    dg = nav.DataGenerator(
        process_model,
        input_profile,
        input_covariance,
        input_freq,
        range_models,
        range_freqs,
    )

    # ##############################################################################
    # Trial function

    gt_data, input_data, meas_data = dg.generate(x0, 0, 50, noise=True)

    x = x0.copy()
    x.plus(nav.randvec(P0))
    x = nav.StateWithCovariance(x, P0)

    ekf = nav.ExtendedKalmanFilter(process_model)

    meas_idx = 0
    y = meas_data[meas_idx]
    results_list = []
    for k in range(len(input_data) - 1):

        # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        # The data generator will add noise on top of the already random signal if 
        # `input_covariance` is not zero. So here we remove this.
        
        u: nav.StampedValue = input_data[k]
        u.value = np.zeros(u.value.shape) # Zero-out the input

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

            x = ekf.correct(x, y, u)

            # Load the next measurement
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        x = ekf.predict(x, u)
        results_list.append(nav.GaussianResult(x, gt_data[k]))

    results = nav.GaussianResultList(results_list)

    return results

if __name__ == "__main__":
    results = main()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2, sharex=True)
    nav.plot_error(results, axs)
    axs[0,0].set_title("Position Error")
    axs[0,1].set_title("Velocity Error")
    axs[1,0].set_xlabel("Time (s)")
    axs[1,1].set_xlabel("Time (s)")
    axs[0,0].set_ylabel("x (m)")
    axs[1,0].set_ylabel("y (m)")
    axs[0,1].set_ylabel("x (m/s)")
    axs[1,1].set_ylabel("y (m/s)")
    plt.tight_layout()
    plt.show()
