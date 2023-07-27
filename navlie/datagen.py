from typing import Callable, List, Union, Any
import numpy as np
from navlie.utils import randvec
from navlie.types import (
    State,
    ProcessModel,
    MeasurementModel,
    Input,
    StampedValue,
    Measurement,
)


class DataGenerator:
    """
    A class used for data generation given a process model, and as many measurement
    models as you want. Frequencies and offsets of each measurement can also be specified.

    Parameters
    ----------
    process_model : ProcessModel
        Process model to be used for ground truth trajectory simulation
    input_func : Callable[[float], np.ndarray]
        A function that returns the input value to be used. The function
        must accept a timestamp as a float from the data generator.
    input_covariance : np.ndarray or Callable[[float], np.ndarray].
        Covariance used for noise generation to be applied to input values.
        Either provided as a static value or as a function returning the time-varying Q
    input_freq : float
        Frequency if the input.
    meas_model_list : List[MeasurementModel], optional
        MeasurementModel of measurements to be generated, by default []
    meas_freq_list : float or List[float], optional
        If a measurement model list is provided, also provide a list of
        frequencies associated with each measurement. If a single frequency
        is provided as a float, it will be used for all measurements.
    meas_offset_list : float or List[float], optional
        If a measurement model list is provided, you can also provide a list of
        initial offsets associated with each measurement. If no offsets provided,
        the initial offset is assumed to be 0. If a single offset is provided
        as a float, it will be used for all measurements.
    """

    def __init__(
        self,
        process_model: ProcessModel,
        input_func: Callable[[float], np.ndarray],
        input_covariance: np.ndarray,
        input_freq: float,
        meas_model_list: List[MeasurementModel] = [],
        meas_freq_list: Union[float, List[float]] = None,
        meas_offset_list: Union[float, List[float]] = [],
    ):
        
        # Make input covariance a callable if it isnt
        if callable(input_covariance):
            self.input_covariance = input_covariance
        elif isinstance(input_covariance, np.ndarray):
            self.input_covariance = lambda t: input_covariance
        else:
            raise ValueError("Input covariance must be a function or a matrix.")

        # Check meas frequencies were provided
        if (len(meas_model_list) != 0) and (meas_freq_list is None):
            raise ValueError("Measurement frequency must be provided.")
        
        # If meas model provided but no offsets, set offsets to 0
        if (len(meas_model_list) != 0) and (len(meas_offset_list) == 0):
            meas_offset_list = [0.0]

        # If only one frequency was provided, assume it was for all the models.
        if not isinstance(meas_freq_list, list):
            meas_freq_list = [meas_freq_list]

        if len(meas_freq_list) == 1:
            meas_freq_list = meas_freq_list * len(meas_model_list)

        # If only one offset was provided, assume it was for all the models.
        if isinstance(meas_offset_list, float):
            meas_offset_list = [meas_offset_list] * len(meas_model_list)
        elif isinstance(meas_offset_list, list):
            if len(meas_offset_list) == 1:
                meas_offset_list *= len(meas_model_list)
            elif len(meas_offset_list) != len(meas_model_list):
                raise ValueError("The number of offsets and \
                                 measurement models should be the same.")
        else:
            raise ValueError("Offsets should be provided as a float or a list of floats.")

        self.process_model = process_model
        self.input_func = input_func
        self.input_freq = input_freq
        self._meas_model_info = list(zip(
            meas_model_list, 
            meas_freq_list, 
            meas_offset_list,
        ))


    def add_measurement_model(
        self, 
        model: MeasurementModel, 
        freq: float, 
        offset: float = 0.0
    ):
        """Adds a new measurement model to the existing list of measurement models. 

        Parameters
        ----------
        model : MeasurementModel
            The MeasurementModel of measurements to be generated.
        freq : float
            The frequency associated with these measurements.
        offset : float, optional
            The initial offset associated with these measurements, by default 0.0.
        """
        self._meas_model_info.append((model, freq, offset))

    def generate(self, x0: State, start: float, stop: float, noise=False):
        """
        Generates data by integrating the process model using noiseless input
        values, and using the states computed to generate measuements.

        Parameters
        ----------
        x0 : State
            Initial state
        start : float
            Starting timestamp
        stop : float
            Ending timestamp
        noise : bool, optional
            Whether to add noise to inputs/measurements, by default False

        Returns
        -------
        List[State]
            Ground truth states
        List[Input]
            Inputs, possibly noisy if requested.
        List[Measurement]
            Measurements, possibly noisy if requested.

        """

        times = np.arange(start, stop, 1 / self.input_freq)
        times = np.round(times, 12)

        # Build large list of Measurement objects with the correct stamps,
        # but empty values, which we will fill later.
        meas_list: List[Measurement] = []
        for model_freq_offset in self._meas_model_info:
            model, freq, offset = model_freq_offset
            stamps = np.arange(times[0] + offset, times[-1], 1 / freq)
            stamps = np.round(stamps, 12)
            temp = [Measurement(None, stamp, model) for stamp in stamps]
            meas_list.extend(temp)

        # Sort by stamp
        meas_list.sort(key=lambda meas: meas.stamp)

        meas_idx = 0
        meas_generated = False

        if meas_idx < len(meas_list):
            meas = meas_list[meas_idx]
        else:
            meas_generated = True

        x = x0.copy()
        x.stamp = times[0]
        state_list = [x.copy()]
        input_list: List[Input] = []

        for k in range(0, len(times) - 1):
            # Check if the provided input profile is an object with a stamp
            # or is just the raw value
            u = self.input_func(times[k], x)
            Q = np.atleast_2d(self.input_covariance(times[k]))

            # If just the raw value, converted to a StampedValue object
            if not hasattr(u, "stamp"):
                u = StampedValue(
                    value=self.input_func(times[k], x),
                    stamp=times[k],
                    covariance=Q,
                )

            # Generate measurements if it is time to do so
            if not meas_generated:
                while times[k + 1] > meas.stamp and not meas_generated:
                    # Propagate state to measurement time
                    dt = meas.stamp - x.stamp
                    x = self.process_model.evaluate(x.copy(), u, dt)
                    x.stamp = meas.stamp

                    # Generate measurement
                    meas.value = generate_measurement(
                        state=x, model=meas.model, noise=noise
                    ).value

                    # Load next measurement
                    meas_idx += 1
                    if meas_idx < len(meas_list):
                        meas = meas_list[meas_idx]
                    else:
                        meas_generated = True

            # Propagate forward
            dt = times[k + 1] - x.stamp
            x = self.process_model.evaluate(x.copy(), u, dt)
            x.stamp = times[k + 1]

            # Add noise to input if requested.
            if noise:
                u = u.plus(randvec(Q))

            state_list.append(x.copy())
            input_list.append(u)

        state_list.sort(key=lambda x: x.stamp)
        input_list.sort(key=lambda x: x.stamp)
        meas_list.sort(key=lambda x: x.stamp)
        return state_list, input_list, meas_list


def generate_measurement(
    state: Union[State, List[State]],
    model: MeasurementModel,
    noise=True,
    state_id: Any = None,
) -> Union[Measurement, List[Measurement]]:
    """
    Generates a `Measurement` object given a measurement model and corresponding
    ground truth `State` objects. If a list of ground truth state objects is
    provided, a measurement will be generated for each state.

    Parameters
    ----------
    state : State or List[State]
        state at which to generate the measurement
    model : MeasurementModel
        measurement model that will be evaluated to generate the measurement
    noise : bool, optional
        flag whether to add noise to measurement, by default True
    state_id : Any, optional
        value to be given to the state_id field of the `Measurement` object

    Returns
    -------
    Measurement or List[Measurement]
        generated Measurement object(s)
    """

    # Handle list or single state
    if isinstance(state, State):
        received_list = False
        x_list = [state]
    else:
        received_list = True
        x_list = state

    # Generate all the measurements
    meas_list = []
    for x in x_list:
        R = np.atleast_2d(model.covariance(x))
        y = model.evaluate(x)
        og_shape = y.shape
        if noise:
            y = y.reshape((-1, 1)) + randvec(R)

        meas_list.append(
            Measurement(y.reshape(og_shape), x.stamp, model, state_id)
        )

    if not received_list:
        meas_list = meas_list[0]

    return meas_list
