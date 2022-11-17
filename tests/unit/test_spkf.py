from pynav.lib.states import MatrixLieGroupState, VectorState
import numpy as np
from pynav.filters import mean_state


np.random.seed(0)

def test_mean_state_vector():
    n_states = 10
    dim = 5
    x_array = []
    weights = np.random.rand(n_states)
    weights = weights / np.sum(weights)
    for i in range(n_states):
        x = VectorState(5*np.random.rand(dim))
        x_array.append(x)

    x_mean_function = mean_state(x_array, weights)    
    x_mean = np.zeros(dim)
    for i in range(n_states):
        x_mean += weights[i] * x_array[i].value
    
    assert np.allclose(x_mean_function.value, x_mean)

if __name__ == "__main__":
    test_mean_state_vector()