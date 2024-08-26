import pytest
from navlie.lib.states import VectorState, SE2State, MixtureState
from navlie.datagen import DataGenerator
from navlie.utils import GaussianResult
from navlie.utils import randvec

from navlie.utils import monte_carlo
from navlie.lib.models import SingleIntegrator, RangePointToAnchor
from navlie.lib.models import (
    BodyFrameVelocity,
)
from navlie.lib.models import RangePoseToAnchor

import numpy as np
from typing import List
from navlie.filters import GaussianSumFilter, run_gsf_filter
from navlie.utils import MixtureResult, MixtureResultList
from navlie.types import StateWithCovariance

def make_range_models_vector(R):
    range_models = [
        RangePointToAnchor([-5, 5], R),
        RangePointToAnchor([ 5, 5], R),
    ]
    return range_models

def make_range_models_se2(R):
    range_models = [
        RangePoseToAnchor([-5, 5],[0, 0], R),
        RangePoseToAnchor([ 5, 5],[0, 0], R),
    ]
    return range_models

def make_filter_trial(dg, x0_true, P0, t_max, gsf, model_states):
    def gsf_trial(trial_number: int) -> List[GaussianResult]:
        """
        A Gaussian Sum Filter trial
        """
        np.random.seed(trial_number)
        state_true, input_list, meas_list = dg.generate(x0_true, 0, t_max, True)

        x = [x_.plus(randvec(P0)) for x_ in model_states]
        x0_check = MixtureState(
            [StateWithCovariance(_x, P0) for _x in x], [1/len(x) for _ in x]
        )

        estimate_list = run_gsf_filter(gsf, x0_check, input_list, meas_list)

        results = [
            MixtureResult(estimate_list[i], state_true[i])
            for i in range(len(estimate_list))
        ]

        return MixtureResultList(results)

    return gsf_trial

@pytest.mark.parametrize(
    "x0, P0, input_profile, Q, process_model, measurement_model, \
        R, input_freq, measurement_freq, model_states",
    [
        (
            VectorState(np.array([0, 0]), stamp=0.0),
            0.1 * np.identity(2),
            lambda t, x: np.array([0.1, np.cos(t)]),
            0.1**2 * np.identity(2),
            SingleIntegrator,
            make_range_models_vector,
            0.1**2,
            10,
            5,
            [
                VectorState(np.array([0, 0]), stamp=0.0),
                VectorState(np.array([0, 10]), stamp=0.0),
            ]
        ),
        (
            SE2State([0, -5, 0], stamp=0.0),
            0.1 * np.identity(3),
            lambda t, x: np.array([0.0, 0.1, np.cos(t)]),
            np.diag([0.01**2, 0.1 ,0.1]),
            BodyFrameVelocity,
            make_range_models_se2,
            0.1**2,
            10,
            5,
            [
                SE2State([0, -5, 0], stamp=0.0),
                SE2State([0,  5, 0], stamp=0.0),
            ],
        ),
    ],
)

def test_reasonable_nees_gsf(
    x0, 
    P0, 
    input_profile, 
    Q, 
    process_model, 
    measurement_model,
    R, 
    input_freq, 
    measurement_freq, 
    model_states
):
    dt = 1 / input_freq
    t_max = dt * 100
    N = 5

    dg = DataGenerator(
        process_model(Q),
        input_profile,
        Q,
        input_freq,
        measurement_model(R),
        measurement_freq,
    )

    gsf_trial = make_filter_trial(
        dg,
        x0,
        P0,
        t_max,
        GaussianSumFilter(process_model(Q)),
        model_states,
    )
    results = monte_carlo(gsf_trial, N)

    nees_in_correct_region = np.count_nonzero(
        results.average_nees < 2 * results.nees_upper_bound(0.99)
    )
    nt = results.average_nees.shape[0]
    # Proportion of time NEES remains below 2*upper_bound bigger than 90%
    assert nees_in_correct_region / nt > 0.90

    # Make sure we essentially never get a completely absurd NEES.
    nees_in_correct_region = np.count_nonzero(
        results.average_nees < 50 * results.nees_upper_bound(0.99)
    )
    assert nees_in_correct_region / nt > 0.9999
