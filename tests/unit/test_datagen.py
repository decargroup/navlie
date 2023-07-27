from navlie.lib.models import SingleIntegrator
from navlie import DataGenerator
import numpy as np


def test_datagen_no_meas_default():
    process_model = SingleIntegrator(Q=np.eye(3))

    try:
        # Create a data generator with default parameter for measurement model list and frequency
        dg = DataGenerator(
            process_model,
            input_func=1,
            input_covariance=np.eye(3),
            input_freq=1,
        )

        # Create a data generator with a blank measurement model list
        dg = DataGenerator(
            process_model,
            input_func=1,
            input_covariance=np.eye(3),
            input_freq=1,
            meas_model_list=[],
        )

    except Exception as e:
        assert False

    else:
        assert True
