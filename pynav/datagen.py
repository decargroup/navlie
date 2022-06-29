from typing import Callable, List
from xml.etree.ElementTree import QName
import numpy as np
from .types import State, ProcessModel, MeasurementModel, StampedValue, Measurement


class DataGenerator:
    def __init__(
        self,
        process_model: ProcessModel,
        input_func: Callable,
        input_covariance: np.ndarray,
        input_freq: float,
        meas_model_list: List[MeasurementModel] = [],
        meas_freq_list: List[float] = [],
    ):
        self.process_model = process_model
        self.input_func = input_func
        self.input_covariance = input_covariance
        self.input_freq = input_freq

        # If only one frequency was provided, assume it was for all the models.
        if len(meas_freq_list) == 1:
            meas_freq_list = meas_freq_list * len(meas_model_list)

        self._meas_model_and_freq = list(zip(meas_model_list, meas_freq_list))

    def add_measurement_model(self, model: MeasurementModel, freq: float):
        self._meas_model_and_freq.append((model, freq))

    def generate(self, x0: State, start:float, stop:float, noise=False):

        times = np.arange(start, stop, 1/self.input_freq)

        # Build large list of Measurement objects with the correct stamps,
        # but empty values.
        meas_list: List[Measurement] = []
        for model_and_freq in self._meas_model_and_freq:
            model, freq = model_and_freq
            stamps = np.arange(times[0], times[-1], 1 / freq)
            temp = [Measurement(None, stamp, model) for stamp in stamps]
            meas_list.extend(temp)

        # Sort by stamp
        meas_list.sort(key=lambda meas: meas.stamp)

        meas_iter = iter(meas_list)
        meas_generated = False

        # Get he first measurement
        try:
            meas = next(meas_iter)
        except StopIteration:
            meas_generated = True
        except Exception as e:
            raise e

        x = x0.copy()
        x.stamp = times[0]
        state_list = [x.copy()]
        input_list: List[StampedValue] = []
        Q = np.atleast_2d(self.input_covariance)
        for i in range(0, len(times) - 1):
            u = StampedValue(self.input_func(times[i]), times[i])

            # Generate measurements if it is time to do so
            if not meas_generated:
                while times[i + 1] > meas.stamp:
                    dt = meas.stamp - times[i]
                    x_meas = self.process_model.evaluate(x.copy(), u, dt)
                    meas.value = meas.model.evaluate(x_meas)

                    # Add noise if requested.
                    if noise:
                        R = np.atleast_2d(meas.model.covariance(x_meas))
                        v: np.ndarray = np.linalg.cholesky(R) @ np.random.normal(
                            0, 1, (meas.value.size, 1)
                        )
                        og_shape = meas.value.shape
                        y_noisy = meas.value.flatten() + v.flatten()
                        meas.value = y_noisy.reshape(og_shape)

                    # Load next measurement
                    try:
                        meas = next(meas_iter)
                    except StopIteration:
                        meas_generated = True
                    except Exception as e:
                        raise e

            # Propagate forward
            dt = times[i + 1] - times[i]
            x = self.process_model.evaluate(x, u, dt)
            x.stamp = times[i + 1]

            # Add noise to input if requested.
            if noise:
                w: np.ndarray = np.linalg.cholesky(Q) @ np.random.normal(
                    0, 1, (u.value.size, 1)
                )
                og_shape = u.value.shape
                u_noisy = u.value.flatten() + w.flatten()
                u.value = u_noisy.reshape(og_shape)


            state_list.append(x.copy())
            input_list.append(u)

        state_list.sort(key=lambda x: x.stamp)
        input_list.sort(key=lambda x: x.stamp)
        meas_list.sort(key=lambda x: x.stamp)
        return state_list, input_list, meas_list


def generate_measurement(x: State, model: MeasurementModel):
    R = np.atleast_2d(model.covariance(x))
    y = model.evaluate(x)
    og_shape = y.shape 
    y_noisy = y + np.linalg.cholesky(R) @ np.random.normal(0,1,(y.size, 1))
    return Measurement(y_noisy.reshape(og_shape), x.stamp, model)

