from pynav.lib.states import VectorState, SO2State, SO3State, SE2State, SE3State, SE23State
from pylie import SO2, SO3, SE2, SE3, SE23
import numpy as np 

def test_plus_minus_vector():
    x1 = VectorState([1,2,3])
    dx = np.array([4,5,6])
    x2 = x1.copy()
    x2.plus(dx) # x2 = x1 + dx 
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_plus_minus_se3():
    x1 = SE3State(SE3.Exp([0.1,0.2,0.3,4,5,6]))
    dx = np.array([0.1, 0.2, 0.4, 0.2, 0.2, 0.2])
    x2 = x1.copy()
    x2.plus(dx) # x2 = x1 + dx 
    dx_test = x2.minus(x1).ravel()
    assert np.allclose(dx, dx_test)

def test_plus_minus_se2():
    x1 = SE2State(SE2.Exp([0.1,5,6]))
    dx = np.array([0.1, 0.2, 0.4])
    x2 = x1.copy()
    x2.plus(dx) # x2 = x1 + dx 
    dx_test = x2.minus(x1).ravel()
    assert np.allclose(dx, dx_test)

def test_plus_minus_so2():
    x1 = SO2State(SO2.Exp([0.1]))
    dx = np.array([0.3])
    x2 = x1.copy()
    x2.plus(dx) # x2 = x1 + dx 
    dx_test = x2.minus(x1).ravel()
    assert np.allclose(dx, dx_test)

def test_plus_minus_so3():
    x1 = SO3State(SO3.Exp([0.1, 0.2, 0.3]))
    dx = np.array([0.3, 0.4, 0.5])
    x2 = x1.copy()
    x2.plus(dx) # x2 = x1 + dx 
    dx_test = x2.minus(x1).ravel()
    assert np.allclose(dx, dx_test)

def test_plus_minus_se23():
    x1 = SE23State(SE23.Exp(0.1*np.array([1,2,3,4,5,6,7,8,9])))
    dx = 0.1*np.array([3,4,5,6,7,8,9,1,2])
    x2 = x1.copy()
    x2.plus(dx) # x2 = x1 + dx 
    dx_test = x2.minus(x1).ravel()
    assert np.allclose(dx, dx_test)

if __name__ == "__main__":
    test_plus_minus_se3()