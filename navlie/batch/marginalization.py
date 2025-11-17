import numpy as np
from scipy.linalg import eigh, solve

from typing import Dict, Hashable, List, Tuple, Callable


from navlie.batch import residuals, losses
from navlie.batch.problem import Problem
from navlie.types import State
from navlie.lib.states import CompositeState


class MarginalizationProblem(Problem):

    def __init__(
        self,
        original_problem: Problem,
        marginalized_keys: List[Hashable],
        sort_keys: Callable,
    ):

        super().__init__()
        self.original_problem = original_problem
        self.marginalized_residuals: List[
            Tuple[residuals.Residual, losses.LossFunction]
        ] = [
            (res, loss)
            for res, loss in zip(
                original_problem.residuals_list, original_problem.losses_list
            )
            if any(key in marginalized_keys for key in res.keys)
        ]

        remain_keys = list(
            {key for res, _ in self.marginalized_residuals for key in res.keys}
        )

        remain_keys.sort(key=sort_keys)

        self.marginalized_variables: Dict[Hashable, State] = {
            key: original_problem.variables[key] for key in remain_keys
        }

        self.remain_variables: Dict[Hashable, State] = {
            key: original_problem.variables[key]
            for key in original_problem.variables
            if key not in marginalized_keys
        }

        self.variables_init: Dict[Hashable, State] = {
            **self.remain_variables,
            **self.marginalized_variables,
        }

        self.variables = {key: val.copy() for key, val in self.variables_init.items()}

        self.constant_variable_keys = original_problem.constant_variable_keys.copy()

    def marginalize(self):

        e_m, H_m, _ = self.compute_error_jac_cost()
        slice_mm = slice(
            (
                self.variable_slices[
                    list(self.marginalized_variables.keys())[
                        len(self.constant_variable_keys)
                    ]
                ].start
                if self.constant_variable_keys
                else 0
            ),
            self.variable_slices[list(self.marginalized_variables.keys())[-1]].stop,
        )
        slice_rr = slice(
            self.variable_slices[list(self.remain_variables.keys())[0]].start,
            self.variable_slices[list(self.remain_variables.keys())[-1]].stop,
        )
        A = H_m.T @ H_m
        b = -H_m.T @ e_m

        Am, bm = compute_schur_complement(A, b, slice_mm, slice_rr)
        marg_info = {
            "Am": Am,
            "bm": bm,
            "prior_states": self.remain_variables.copy(),
        }

        return marg_info


def compute_schur_complement(
    A: np.ndarray,
    b: np.ndarray,
    slice_mm: slice,
    slice_rr: slice,
    eps: float = 1e-8,
):

    A_mm = A[slice_mm, slice_mm]
    A_mr = A[slice_mm, slice_rr]
    A_rm = A[slice_rr, slice_mm]
    A_rr = A[slice_rr, slice_rr]
    b_mm = b[slice_mm]
    b_rr = b[slice_rr]
    A_mm = (A_mm + A_mm.T) / 2

    A_mm_inv = solve(
        A_mm,
        np.identity(A_mm.shape[0]),
        overwrite_a=True,
        overwrite_b=True,
        assume_a="pos",
        check_finite=False,
    )
    # A_mm_inv = np.linalg.inv(A_mm)
    bp = b_rr - A_rm @ A_mm_inv @ b_mm
    Ap = A_rr - A_rm @ A_mm_inv @ A_mr
    # Ap = Ap.toarray()
    # eigendecomposition for cholesky
    # w, V = np.linalg.eigh(Ap)
    w, V = eigh(Ap, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False)
    V: np.ndarray = np.real(V)
    w = np.real(w)
    w_num = np.where(w > eps, w, eps)
    w_inv = np.where(w_num > eps, 1 / w_num, 0)
    sqrt_w = np.sqrt(w_num)
    sqrt_w_inv = np.sqrt(w_inv)
    Am = np.diag(sqrt_w) @ V.T
    bm = np.diag(sqrt_w_inv) @ V.T @ bp

    return Am, bm


class MarginalizedPriorResidual(residuals.Residual):
    def __init__(
        self,
        marginalization_info: Dict[str, np.ndarray, Dict[Hashable, State]],
    ):

        keys = list(marginalization_info["prior_states"].keys())
        self._Am = marginalization_info["Am"]
        self._bm = marginalization_info["bm"]
        self._prior_states = marginalization_info["prior_states"]
        super().__init__(keys)

    def evaluate(self, states, compute_jacobian=None):
        return super().evaluate(states, compute_jacobian)

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:

        x_p = CompositeState(self._prior_states)
        x = CompositeState(states)
        error = self._Am @ x.minus(x_p) + self._bm.ravel()

        if compute_jacobians:
            jacobians = [None] * len(states)
            col_slice = [0, 0]
            for i, state in enumerate(states):
                col_slice[0] = col_slice[1]
                col_slice[1] += state.dof
                if compute_jacobians[i]:
                    jacobians[i] = self._Am[
                        :, col_slice[0] : col_slice[1]
                    ]  # each column is a jacobian

            return error, jacobians

        return error
