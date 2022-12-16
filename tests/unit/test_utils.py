from pynav.utils import GaussianResult, GaussianResultList
from pynav.types import StateWithCovariance
from pynav.lib.states import SE23State
from pylie import SE23
import numpy as np

def test_gaussian_result_indexing():
    x_true = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    x_hat = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    cov = [(i+1)*np.eye(9) for i in range(10)]
    x_cov = [StateWithCovariance(x, c) for x, c in zip(x_hat, cov)]
    gr = [GaussianResult(x, t) for x, t in zip(x_cov, x_true)]
    grl = GaussianResultList(gr)


    slc = 2
    grl_test = grl[:,slc]
    e_test = grl.error[:,slc]
    cov_test = grl.covariance[:, slc, slc]
    nees_test =  [e**2/c for e, c in zip(e_test, cov_test)]
    nees_test = np.array(nees_test).squeeze()
    assert np.alltrue(grl_test.covariance == cov_test)
    assert np.alltrue(grl_test.error == grl.error[:,slc])
    assert np.allclose(grl_test.nees, nees_test)

def test_gaussian_result_slicing():
    x_true = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    x_hat = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    cov = [(i+1)*np.eye(9) for i in range(10)]
    x_cov = [StateWithCovariance(x, c) for x, c in zip(x_hat, cov)]
    gr = [GaussianResult(x, t) for x, t in zip(x_cov, x_true)]
    grl = GaussianResultList(gr)


    slc = slice(0,3)
    grl_test = grl[:,slc]
    e_test = grl.error[:,slc]
    cov_test = grl.covariance[:, slc, slc]
    nees_test = [e.reshape((1,-1)) @ np.linalg.inv(c) @ e.reshape((-1,1)) for e, c in zip(e_test, cov_test)]
    nees_test = np.array(nees_test).squeeze()
    assert np.alltrue(grl_test.covariance == cov_test)
    assert np.alltrue(grl_test.error == grl.error[:,slc])
    assert np.allclose(grl_test.nees, nees_test)


if __name__ == '__main__':
    test_gaussian_result_indexing()