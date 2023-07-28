"""
This is an example script showing how to generate measurements using multiple
measurement models that are supposed to run in sequence rather than in parallel.
The example shown here is a transceiver communicating with 3 anchors, where it
can only communicate with one anchor at a time. We therefore loop through the
MeasurementModels to generate a measurement to one anchor at every time instance.
"""

from navlie.lib.states import VectorState
from navlie.datagen import DataGenerator
from navlie.utils import schedule_sequential_measurements
from navlie.lib.models import SingleIntegrator, RangePointToAnchor
import numpy as np
import matplotlib.pyplot as plt

def main():

    # ##############################################################################
    # Problem Setup

    x0 = VectorState(np.array([1, 0]), stamp=0)
    P0 = np.diag([1, 1])
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_models = [
        RangePointToAnchor([0, 4], R), # Anchor 1
        RangePointToAnchor([-2, 0], R), # Anchor 2
        RangePointToAnchor([2, 0], R), # Anchor 3
    ]
    range_freqs = 100 # This defines the overall frequency in which we want the
                    # frequencies of the 3 measurement models to sum up to.
    process_model = SingleIntegrator(Q)
    input_profile = lambda t, x: np.array([0, 0])
    input_covariance = Q
    input_freq = 180
    noise_active = True

    # ##############################################################################
    # Schedule sequential measurements

    range_offset, sequential_freq = schedule_sequential_measurements(
        range_models, range_freqs
    )

    # ##############################################################################
    # Data Generation

    dg = DataGenerator(
        process_model,
        input_profile,
        input_covariance,
        input_freq,
        range_models,
        sequential_freq, # reduced frequency of each individual MeasurementModel
        range_offset, # each measurement is offset so they do not start at the same time
    )

    _, _, meas_data = dg.generate(x0, 0, 0.1)

    print("The effective frequency of every measurements to every Anchor is " \
        + str(sequential_freq) + " Hz.")

    return meas_data

if __name__ == "__main__":
    meas_data = main()
    # We can then see in the plot that the measurement to each anchor is being generated 
    # sequentially with no overlap. 
    plt.scatter([x.stamp for x in meas_data], [x.value for x in meas_data])
    plt.grid()
    plt.ylabel(r"Measurement Value [m]")
    plt.xlabel(r"Time [s]")
    plt.show()