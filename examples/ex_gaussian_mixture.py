import matplotlib.pyplot as plt
import numpy as np

from navlie.batch.gaussian_mixtures import (
    GaussianMixtureResidual,
    HessianSumMixtureResidual,
    MaxMixtureResidual,
    MaxSumMixtureResidual,
    SumMixtureResidual,
)
from navlie.batch.problem import Problem
from navlie.batch.residuals import PriorResidual
from navlie.lib.states import VectorState


def main():
    key = "x"
    component_residuals = []
    stamp = 0.0
    means = [np.array([0.0]), np.array([0.5]), np.array([1])]
    covariances = [
        np.atleast_2d(np.array([2])),
        np.atleast_2d(np.array([2])),
        np.atleast_2d(np.array([3])),
    ]
    weights = [0.5, 0.5]
    for lv1 in range(len(means)):
        prior_state = VectorState(means[lv1], stamp)

        component_residuals.append(PriorResidual([key], prior_state, covariances[lv1]))
    res_dict = {
        "Max-Mixture": MaxMixtureResidual(component_residuals, weights),
        "Sum-Mixture": SumMixtureResidual(component_residuals, weights),
        "Max-Sum-Mixture": MaxSumMixtureResidual(component_residuals, weights, 10),
        "Hessian-Sum-Mixture": HessianSumMixtureResidual(
            component_residuals, weights, True
        ),
    }

    x0 = VectorState(2, 0.0, "x")
    plt.figure()
    for key, res in res_dict.items():
        x = x0.copy()
        print(f"Running {key} optimization...")
        problem = Problem(
            solver="LM",
            max_iters=100,
            step_tol=1e-8,
            tau=1e-11,
            verbose=False,
        )
        problem.add_residual(res)
        problem.add_variable("x", x)
        opt_nv_res = problem.solve()
        x = np.linspace(-3, 3, 1000)
        linestyles = ["-", "--", "-.", ":", "-"] * 6

        res: MaxMixtureResidual = res
        plt.plot(
            x,
            np.array([evaluate_log_likelihood(res, val) for val in x]),
            label=key,
            linestyle=linestyles[lv1],
        )
        plt.scatter(
            opt_nv_res["variables"]["x"].value,
            np.sum(res.evaluate([opt_nv_res["variables"]["x"]]) ** 2),
            marker="o",
        )
    plt.xlabel("x")
    plt.ylabel("Cost Function Value (offset by norm. constants)")
    plt.legend()
    plt.show()


def evaluate_error(res: GaussianMixtureResidual, val: np.ndarray):
    error = res.evaluate([VectorState(val.squeeze())])
    error = np.atleast_1d(error)
    return error


def evaluate_log_likelihood(res: GaussianMixtureResidual, val: np.ndarray):
    error = evaluate_error(res, np.atleast_1d(val))
    return np.linalg.norm(error, 2) ** 2


if __name__ == "__main__":
    main()
