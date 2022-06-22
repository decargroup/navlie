from typing import Callable, List
import numpy as np
from .types import State, ProcessModel, MeasurementModel, StampedValue, Measurement


class DataGenerator:
    def __init__(
        self,
        process_model: ProcessModel,
        input_func: Callable,
        meas_model_list: List[MeasurementModel] = [],
        meas_freq_list: List[float] = [],
    ):
        self.process_model = process_model
        self.input_func = input_func

        if len(meas_freq_list) == 1:
            meas_freq_list = meas_freq_list * len(meas_model_list)

        self._meas_model_and_freq = []
        for i in range(len(meas_model_list)):
            model = meas_model_list[i]
            freq = meas_freq_list[i]
            self._meas_model_and_freq.append((model, freq))

    def add_measurement_model(self, model: MeasurementModel, freq: float):
        self._meas_model_and_freq.append((model, freq))

    def generate(self, x0: State, times: np.ndarray, noise=False):

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
        for i in range(0, len(times) - 1):
            u = StampedValue(self.input_func(times[i]), times[i])

            if times[i + 1] > meas.stamp and not meas_generated:
                dt = meas.stamp - times[i]
                x_meas = self.process_model.evaluate(x.copy(), u, dt)
                meas.value = meas.model.evaluate(x_meas)
                try:
                    meas = next(meas_iter)
                except StopIteration:
                    meas_generated = True
                except Exception as e:
                    raise e

            dt = times[i + 1] - times[i]
            x = self.process_model.evaluate(x, u, dt)
            x.stamp = times[i + 1]
            state_list.append(x.copy())
            input_list.append(u)

        state_list.sort(key=lambda x: x.stamp)
        input_list.sort(key=lambda x: x.stamp)
        meas_list.sort(key=lambda x: x.stamp)
        return state_list, input_list, meas_list
