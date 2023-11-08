"""Robust loss functions to be used in nonlinear least squares problems.

Robust loss functions here must inherit from the LossFunction interface, 
which defines a loss and a weight.
"""

from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    """
    Abstract base class for any loss function.
    """

    @abstractmethod
    def loss(self, e: float):
        """
        The loss function defines the cost :math:`\\rho(e)`, where :math:`e` is an
        error term and is often a function of the design variable.
        """
        pass

    @abstractmethod
    def weight(self, e: float):
        """
        The weight that depends on the current value of the error.
        This is used to reweight our original nonlinear least squares problem.
        """
        pass


class L2Loss(LossFunction):
    """
    Standard L2 loss. Simply 0.5 * e * e , with the robust
    weight of ones.
    """

    def loss(self, e: float):
        return 0.5 * e * e

    def weight(self, x: float):
        return 1.0


class CauchyLoss(LossFunction):
    def __init__(self, c: float = 1.0):
        self.c = c

    def loss(self, e: float) -> float:
        """
        Cauchy loss function.

        The form here is taken from "MacTavish, Barfoot - At All Costs."

        Parameters
        ----------
        e : float
            Residual

        Returns
        -------
        float
            Cost evaluated at a given error.
        """
        return (0.5 * self.c**2) * np.log(1.0 + (e / self.c) ** 2)

    def weight(self, e: float) -> float:
        """
        Cauchy weight function.

        Parameters
        ----------
        e : float
            Residual

        Returns
        -------
        float
            Robust weight
        """
        return 1.0 / (1.0 + (e / self.c) ** 2)

class GeneralAdaptiveLoss(LossFunction):
    def __init__(self, alpha: float, c: float):
        self.alpha = alpha
        self.c = c

    def loss(self, e: float) -> float:
        # alpha approaches 2 - L2 Loss
        if (self.alpha == 2.0):
            return 0.5 * (e / self.c)**2
        # alpha approaches 0 - Cauchy Loss
        elif (np.isclose(self.alpha, 0.0)):
            return np.log(1.0 + 0.5 * (e / self.c)**2)
        # alpha approaches -infty - Weslch loss
        elif (self.alpha < -1e10):
            return 1 - np.exp(-0.5 * (e / self.c)**2)
        # The general cost function
        else:
            term_1 = (abs(self.alpha - 2.0) / self.alpha)
            term_2 = ((e / self.c)**2.0 / (abs(self.alpha - 2.0)) + 1)**(self.alpha / 2.0) - 1.0
            cost =  term_1 * term_2
            return cost
    
    def weight(self, e: float) -> float:
        if (self.alpha == 2.0):
            return 1 / self.c**2
        if (self.alpha == 0.0):
            return 2 / (e**2 + 2*self.c**2)
        if (self.alpha < 1e-10):
            return (1 / self.c**2) * np.exp(-0.5 * (e / self.c)**2)
        else:
            term_1 = 1 / self.c**2
            term_2 = (((e / self.c) **2) / (abs(self.alpha - 2.0)) + 1.0)**(self.alpha/2.0 - 1.0)
            return term_1 * term_2