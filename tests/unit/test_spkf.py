from navlie.lib.states import MatrixLieGroupState, VectorState, SE3State
import numpy as np
from navlie.filters import mean_state, generate_sigmapoints
from pymlg import SE3
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

def test_generate_sigmapoints():
    np.random.seed(0)
    x = SE3State(SE3.random())
    sps, w = generate_sigmapoints(x.dof, 'unscented')

    x_propagated = [
                    x.plus(sp)
                for sp in sps.T
            ]

    assert np.allclose(mean_state(x_propagated, w).value, x.value)
    sps, w = generate_sigmapoints(x.dof, 'cubature')

    x_propagated = [
                    x.plus(sp)
                for sp in sps.T
            ]

    assert np.allclose(mean_state(x_propagated, w).value, x.value)
    sps, w = generate_sigmapoints(x.dof, 'gh')

    x_propagated = [
                    x.plus(sp)
                for sp in sps.T
            ]

    assert np.allclose(mean_state(x_propagated, w).value, x.value)




if __name__ == "__main__":
    test_generate_sigmapoints()