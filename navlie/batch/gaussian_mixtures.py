from typing import Hashable, List, Tuple
import numpy as np
from navlie import State
from navlie.batch.residuals import Residual
from abc import ABC, abstractmethod
from typing import Dict


class GaussianMixtureResidual(Residual, ABC):
    """A Gaussian mixture residual.
    Gaussian mixtures can, for instance, model non-Gaussian noise where a mixture has been fit to it, unknown data associations,
    or loop closures where one component corresponds to a true loop closure (small covariance) and
    another component corresponds to a false loop closure (large covariance).

    Implements problem terms of the form

    .. math::
        J(\mathbf{x}) = -\log \sum_{k=1}^{K} w_k \det\left(\sqrt{\mathbf{R}_k^{-1}} \\right)
        \exp (\\boldsymbol{\eta}_k^T (\mathbf{x}) \mathbf{R}_k^{-1} \\boldsymbol{\eta}_k (\mathbf{x}))

    where defining the normalized error :math:`\mathbf{e}_k (\mathbf{x}=\sqrt{\mathbf{R}_k^{-1}} \\boldsymbol{\eta}_k (\mathbf{x})`
    yields

    .. math::
        J(\mathbf{x}) = -\log \sum_{k=1}^{K} w_k \det\left(\sqrt{\mathbf{R}_k^{-1}} \\right)  \exp (\mathbf{e}_k^T (\mathbf{x}) \mathbf{e}_k (\mathbf{x})).

    The errors argument input to the constructor define the errors :math:`\mathbf{e}_k (\mathbf{x})` and the weights define the weights :math:`w_k`.
    The error must define the sqrt_info_matrix method that returns the square root of the information matrix :math:`\sqrt{\mathbf{R}_k^{-1}}`.

    Each type of mixture differs in how the overall error (Jacobian) corresponding to the gaussian mixture is constructed
    from the component errors (component Jacobians). Therefore, each subclass of GaussianMixtureResidual
    must overwrite the mix_errors and mix_jacobians methods.
    """

    def __init__(self, errors: List[Residual], weights: List[float]):
        """
        Parameters
        ----------
        errors : List[Residual]
            List of Residuals, each of which must implement the sqrt_info_matrix method.
        weights : List[float]
            Weights of the Gaussian mixture.
        """
        # Errors that the mixture holds on to, each with its own state keys
        self.errors: List[Residual] = errors

        # Keys of states that get passed into evaluate method
        self.keys: List[Hashable] = []

        # Weights of the Gaussians
        for error in errors:
            for key in error.keys:
                if key not in self.keys:
                    self.keys.append(key)
        self.weights: List[float] = weights / np.sum(np.array(weights))

        self.sqrt_info_matrix_list: List[np.ndarray] = None

    @abstractmethod
    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray]:
        """Each mixture must implement this method..
        Compute the factor error from the errors corresponding to each component

        All errors are assumed to be normalized and have identity covariance.

        Parameters
        ----------
        error_value_list : List[np.ndarray],
            List of errors corresponding to each component
        """
        pass

    @abstractmethod
    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[List[np.ndarray]]:
        """Each mixture must implement this method.
        For every state, compute Jacobian of the Gaussian mixture w.r.t. that state

        Parameters
        ----------
        error_value_list : List[np.ndarray],
            List of errors corresponding to each component
        jacobian_list : List[List[np.ndarray]]
            Outer list corresponds to each component, for each of which the inner list contains
            the component Jacobians w.r.t. every state.
        """
        pass

    def evaluate_component_residuals(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[np.ndarray]]:
        error_value_list: List[np.ndarray] = []
        jacobian_list_of_lists: List[List[np.ndarray]] = []
        sqrt_info_matrix_list: List[np.ndarray] = []

        for error in self.errors:
            cur_keys = error.keys
            key_indices = [self.keys.index(cur_key) for cur_key in cur_keys]
            cur_states = [states[key_idx] for key_idx in key_indices]

            if compute_jacobians:
                # The error and jacobians returned by the sub-error.
                cur_compute_jacobians = [
                    compute_jacobians[key_idx] for key_idx in key_indices
                ]
                val, jac_list_subset = error.evaluate(cur_states, cur_compute_jacobians)
                n_e = val.shape[0]  # Error dimension.

                # Jacobians of states that are not to be computed are set to zero.
                # Jacobians of states that are to be computed, but
                # the state of which is not one the error depends on,
                # are set to zero.
                jac_list_all_states = [None for lv1 in range(len(states))]

                # Set relevant Jacobians to zero first. Then
                # overwrite those that the error depends on.
                for lv1, (compute_jac, state) in enumerate(
                    zip(compute_jacobians, states)
                ):
                    if compute_jac:
                        jac_list_all_states[lv1] = np.zeros((n_e, state.dof))

                # jac_list_subset only has elements corresponding to the states that the error
                # is dependent on.
                # We need to put them in the right place in the list of jacobians
                # that correspond to the whole state list..
                for key_idx, jac in zip(key_indices, jac_list_subset):
                    jac_list_all_states[key_idx] = jac

                jacobian_list_of_lists.append(jac_list_all_states)
            else:
                val = error.evaluate(cur_states)

            error_value_list.append(val)
            sqrt_info_matrix_list.append(error.sqrt_info_matrix(cur_states))
        self.sqrt_info_matrix_list = sqrt_info_matrix_list

        # For the not NLS-compatible HSM version, these values need to be reused for Hessian computation.
        self.error_value_list_cache = error_value_list
        self.jacobian_list_of_lists_cache = jacobian_list_of_lists
        self.sqrt_info_matrix_list_cache = sqrt_info_matrix_list

        return error_value_list, jacobian_list_of_lists, sqrt_info_matrix_list

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        (
            error_value_list,
            jacobian_list_of_lists,
            sqrt_info_matrix_list,
        ) = self.evaluate_component_residuals(states, compute_jacobians)
        e, reused_values = self.mix_errors(error_value_list, sqrt_info_matrix_list)
        if compute_jacobians:
            jac_list = self.mix_jacobians(
                error_value_list,
                jacobian_list_of_lists,
                sqrt_info_matrix_list,
                reused_values,
            )
            return e, jac_list
        return e


class MaxMixtureResidual(GaussianMixtureResidual):
    """
    Based on the following reference,
    @article{olson2013inference,
    title={Inference on networks of mixtures for robust robot mapping},
    author={Olson, Edwin and Agarwal, Pratik},
    journal={The International Journal of Robotics Research},
    volume={32},
    number={7},
    pages={826--840},
    year={2013},
    publisher={SAGE Publications Sage UK: London, England}
    }
    """

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, Dict]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]

        # Maximum component obtained as
        # K = argmax alpha_k exp(-0.5 e^\trans e)
        #   = argmin -2* log alpha_k + e^\trans e
        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * e.T @ e
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        dominant_idx = np.argmin(res_values)
        linear_part = error_value_list[dominant_idx]

        alpha_k = alphas[dominant_idx]
        alpha_max = max(alphas)

        nonlinear_part = np.array(np.log(alpha_max / alpha_k)).reshape(-1)
        nonlinear_part = np.sqrt(2) * np.sqrt(nonlinear_part)
        e_mix = np.concatenate([linear_part, nonlinear_part])

        reused_values = {"alphas": alphas, "res_values": res_values}

        return e_mix, reused_values

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
        reused_values: Dict = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if reused_values is not None:
            alphas = reused_values["alphas"]
            res_values = reused_values["res_values"]
        else:
            alphas = [
                weight * np.linalg.det(sqrt_info_matrix)
                for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
            ]
            res_values = np.array(
                [
                    -np.log(alpha) + 0.5 * e.T @ e
                    for alpha, e in zip(alphas, error_value_list)
                ]
            )
        dominant_idx = np.argmin(res_values)
        jac_list_linear_part: List[np.ndarray] = jacobian_list_of_lists[dominant_idx]

        jac_list = []
        for jac in jac_list_linear_part:
            if jac is not None:
                jac_list.append(np.vstack([jac, np.zeros((1, jac.shape[1]))]))
            else:
                jac_list.append(None)
        return jac_list


class MaxSumMixtureResidual(GaussianMixtureResidual):
    """
    Based on the following reference:
    @ARTICLE{9381625,
    author={Pfeifer, Tim and Lange, Sven and Protzel, Peter},
    journal={IEEE Robotics and Automation Letters},
    title={Advancing Mixture Models for Least Squares Optimization},
    year={2021},
    volume={6},
    number={2},
    pages={3941-3948},
    doi={10.1109/LRA.2021.3067307}}
    """

    damping_const: float

    def __init__(self, errors: List[Residual], weights, damping_const: float = 10):
        super().__init__(errors, weights)
        self.damping_const = damping_const

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]

        # Linear part is the same as for the max-mixture
        # Maximum component obtained as
        # K = argmax alpha_k exp(-0.5 e^\trans e)
        #   = argmin -2* log alpha_k + e^\trans e
        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * e.T @ e
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        dominant_idx = np.argmin(res_values)
        linear_part = error_value_list[dominant_idx]

        # Nonlinear part changes quite a bit. Very similar to sum-mixture.
        err_kmax = error_value_list[dominant_idx]
        scalar_errors_differences = [
            -0.5 * e.T @ e + 0.5 * err_kmax.T @ err_kmax for e in error_value_list
        ]

        nonlinear_part = self.compute_nonlinear_part(scalar_errors_differences, alphas)
        e_mix = np.concatenate([linear_part, nonlinear_part])

        reused_values = {
            "alphas": alphas,
            "nonlinear_part": nonlinear_part,
            "scalar_errors_differences": scalar_errors_differences,
            "res_values": res_values,
        }

        return e_mix, reused_values

    def compute_nonlinear_part(
        self, scalar_errors_differences: List[np.ndarray], alphas: List[float]
    ):
        alpha_max = max(alphas)
        normalization_const = np.log(len(alphas) * alpha_max + self.damping_const)

        sum_term = np.log(
            np.sum(
                np.array(
                    [
                        alpha * np.exp(e)
                        for alpha, e in zip(alphas, scalar_errors_differences)
                    ]
                )
            )
        )
        nonlinear_part = np.sqrt(2) * np.sqrt(normalization_const - sum_term)
        nonlinear_part = np.array(nonlinear_part).reshape(-1)
        return nonlinear_part

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
        reused_values: Dict = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_state_list = len(jacobian_list_of_lists[0])

        if reused_values is not None:
            alphas = reused_values["alphas"]
            scalar_errors_differences = reused_values["scalar_errors_differences"]
            e_nl = reused_values["nonlinear_part"]
            res_values = reused_values["res_values"]
        else:
            alphas = [
                weight * np.linalg.det(sqrt_info_matrix)
                for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
            ]
            # LINEAR PART
            res_values = np.array(
                [
                    -np.log(alpha) + 0.5 * e.T @ e
                    for alpha, e in zip(alphas, error_value_list)
                ]
            )

            scalar_errors_differences = [
                -0.5 * e.T @ e + 0.5 * err_kmax.T @ err_kmax for e in error_value_list
            ]

            # NONLINEAR PART
            # Compute error
            e_nl = self.compute_nonlinear_part(scalar_errors_differences, alphas)

        dominant_idx = np.argmin(res_values)
        err_kmax = error_value_list[dominant_idx]
        jac_list_linear_part: List[np.ndarray] = jacobian_list_of_lists[dominant_idx]
        # Loop through every state to compute Jacobian with respect to it.
        jac_list_nl = []
        denominator_list = [
            alpha * np.exp(scal_err)
            for alpha, scal_err in zip(alphas, scalar_errors_differences)
        ]

        denominator = 0.0
        for term in denominator_list:
            denominator += term
        denominator = denominator * e_nl

        for lv1 in range(n_state_list):
            jacobian_list_components_wrt_cur_state = [
                jac_list[lv1] for jac_list in jacobian_list_of_lists
            ]
            if jacobian_list_components_wrt_cur_state[0] is not None:
                jac_dom = jacobian_list_components_wrt_cur_state[dominant_idx]
                n_x = jacobian_list_components_wrt_cur_state[0].shape[1]
                numerator = np.zeros((1, n_x))

                numerator_list = [
                    -alpha
                    * np.exp(scal_err)
                    * (
                        e_k.reshape(1, -1) @ -jac_e_i
                        + err_kmax.reshape(1, -1) @ jac_dom
                    )
                    for alpha, scal_err, e_k, jac_e_i in zip(
                        alphas,
                        scalar_errors_differences,
                        error_value_list,
                        jacobian_list_components_wrt_cur_state,
                    )
                ]

                for term in numerator_list:
                    numerator += term

                jac_list_nl.append(numerator / denominator)
            else:
                jac_list_nl.append(None)

        jac_list = []
        for jac_lin, jac_nl in zip(jac_list_linear_part, jac_list_nl):
            if jac_nl is not None:
                jac_list.append(np.vstack([jac_lin, jac_nl]))
            else:
                jac_list.append(None)

        return jac_list


class SumMixtureResidual(GaussianMixtureResidual):
    """
    For details see the reference
    @ARTICLE{9381625,
    author={Pfeifer, Tim and Lange, Sven and Protzel, Peter},
    journal={IEEE Robotics and Automation Letters},
    title={Advancing Mixture Models for Least Squares Optimization},
    year={2021},
    volume={6},
    number={2},
    pages={3941-3948},
    doi={10.1109/LRA.2021.3067307}}
    """

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        normalization_const = sum(alphas)

        # K = argmax alpha_k exp(-0.5 e^\trans e)
        #   = argmin -2* log alpha_k + e^\trans e
        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * np.linalg.norm(e) ** 2
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        kmax = np.argmin(res_values)

        scalar_errors = np.array(
            [0.5 * np.linalg.norm(e) ** 2 for alpha, e in zip(alphas, error_value_list)]
        )

        sum_term = np.log(
            np.sum(
                np.array(
                    [
                        alpha * np.exp(-e + scalar_errors[kmax])
                        for alpha, e in zip(alphas, scalar_errors)
                    ]
                )
            )
        )
        e = np.sqrt(2) * np.sqrt(normalization_const + scalar_errors[kmax] - sum_term)
        reused_values = {"alphas": alphas, "scalar_errors": scalar_errors, "e_sm": e}
        return e, reused_values

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[
            List[np.ndarray]
        ],  # outer list is components, inner list states
        sqrt_info_matrix_list: List[np.ndarray],
        reused_values: Dict = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if reused_values is not None:
            alpha_list = reused_values["alphas"]
            e_sm = reused_values["e_sm"]
        else:
            alpha_list = [
                weight * np.linalg.det(sqrt_info_matrix)
                for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
            ]
            e_sm, _ = self.mix_errors(error_value_list, sqrt_info_matrix_list)

        n_state_list = len(jacobian_list_of_lists[0])

        error_value_list = [e.reshape(-1, 1) for e in error_value_list]
        eTe_list = [e.T @ e for e in error_value_list]
        eTe_dom = min(eTe_list)
        exp_list = [np.exp(-0.5 * e.T @ e + 0.5 * eTe_dom) for e in error_value_list]
        sum_exp = np.sum(
            [
                alpha * np.exp(-0.5 * e.T @ e + 0.5 * eTe_dom)
                for alpha, e in zip(alpha_list, error_value_list)
            ]
        )

        drho_df_list = [
            alpha * exp / sum_exp for alpha, exp in zip(alpha_list, exp_list)
        ]

        # Loop through every state to compute Jacobian with respect to it.
        jac_list = []
        for lv1 in range(n_state_list):
            jacobian_list_components_wrt_cur_state = [
                jac_list[lv1] for jac_list in jacobian_list_of_lists
            ]
            if jacobian_list_components_wrt_cur_state[0] is not None:
                n_x = jacobian_list_components_wrt_cur_state[0].shape[1]

                f_i_jac_list = [
                    e.T @ dedx
                    for e, dedx in zip(
                        error_value_list, jacobian_list_components_wrt_cur_state
                    )
                ]

                numerator = np.zeros((1, n_x))

                numerator_list = [
                    drho * f_i_jac for drho, f_i_jac in zip(drho_df_list, f_i_jac_list)
                ]
                for term in numerator_list:
                    numerator += term
                numerator = numerator
                denominator = e_sm
                jac_list.append(numerator / denominator)
            else:
                jac_list.append(None)
        return jac_list


class HessianSumMixtureResidual(GaussianMixtureResidual):
    """
    The Hessian-Sum-Mixture method patched for compatibility with nonlinear least squares solvers.
    Based on the following reference:
    @misc{korotkine2024hessian,
    title={A Hessian for Gaussian Mixture Likelihoods in Nonlinear Least Squares},
    author={Vassili Korotkine and Mitchell Cohen and James Richard Forbes},
    year={2024},
    eprint={2404.05452},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
    }
    """

    no_use_complex_numbers: bool

    def __init__(
        self,
        errors: List[Residual],
        weights,
        no_use_complex_numbers=True,
    ):
        super().__init__(errors, weights)
        self.sum_mixture_residual = SumMixtureResidual(errors, weights)
        self.no_use_complex_numbers = no_use_complex_numbers

    @staticmethod
    def get_normalization_constant(alphas: List[float]):
        alpha_sum = np.sum(alphas)
        log_sum = 0.0
        for lv1 in range(len(alphas)):
            log_sum = log_sum + alphas[lv1] * np.exp(alpha_sum / alphas[lv1])

        return np.log(log_sum)

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        alpha_list = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        normalization_constant = self.get_normalization_constant(alpha_list)

        f_list = [0.5 * np.sum(e**2) for e in error_value_list]
        kmax = np.argmin(np.array(f_list))
        f_kmax = f_list[kmax]
        sum_exp = np.sum(
            [alpha * np.exp(f_kmax - f) for alpha, f in zip(alpha_list, f_list)]
        )

        drho_df_list = [
            alpha * np.exp(f_kmax - f) / sum_exp for alpha, f in zip(alpha_list, f_list)
        ]

        hsm_error = np.vstack(
            [
                np.sqrt(drho) * e.reshape(-1, 1)
                for drho, e in zip(drho_df_list, error_value_list)
            ]
        ).squeeze()

        # When the loss is computed at the end, it is computed as 1/2 * e^\trans e.
        # The normalization constant is a bound on 2*logsumexp minus the norm of hsm_error.
        # This works out to at the end evaluate normalization_constant + f_kmax - np.log(sum_exp).
        desired_loss = 2 * (normalization_constant + f_kmax - np.log(sum_exp))

        if not self.no_use_complex_numbers:
            current_loss = np.sum(hsm_error**2)
            diff = np.array(np.emath.sqrt(desired_loss - current_loss))
            hsm_error = np.concatenate(
                [
                    hsm_error,
                    np.atleast_1d(np.array(diff)),
                ]
            )
        if self.no_use_complex_numbers:
            current_loss = np.sum(hsm_error**2)
            delta = desired_loss - current_loss

            diff = np.array(np.sqrt(delta))
            hsm_error = np.concatenate(
                [
                    hsm_error,
                    np.atleast_1d(np.array(diff)),
                ]
            )
        reused_values = {
            "alphas": alpha_list,
            "f_list": f_list,
            "sum_exp": sum_exp,
            "normalization_constant": normalization_constant,
            "sum_exp": sum_exp,
            "drho_df_list": drho_df_list,
        }
        return hsm_error, reused_values

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
        reused_values: Dict = None,
    ) -> List[np.ndarray]:
        n_state_list = len(jacobian_list_of_lists[0])
        if reused_values is not None:
            drho_df_list = reused_values["drho_df_list"]
        else:
            alpha_list = [
                weight * np.linalg.det(sqrt_info_matrix)
                for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
            ]
            error_value_list = [e.reshape(-1, 1) for e in error_value_list]
            eTe_list = [e.T @ e for e in error_value_list]

            # Normalize all the exponent arguments to avoid numerical issues.
            eTe_dom = min(eTe_list)
            sum_exp = np.sum(
                [
                    alpha * np.exp(0.5 * eTe_dom - 0.5 * e.T @ e)
                    for alpha, e in zip(alpha_list, error_value_list)
                ]
            )

            drho_df_list = [
                alpha * np.exp(0.5 * eTe_dom - 0.5 * eTe) / sum_exp
                for alpha, eTe in zip(alpha_list, eTe_list)
            ]

        jac_list = []
        for lv1 in range(n_state_list):
            jacobian_list_components_wrt_cur_state = [
                jac_list[lv1] for jac_list in jacobian_list_of_lists
            ]
            if jacobian_list_components_wrt_cur_state[0] is not None:
                nx = jacobian_list_components_wrt_cur_state[0].shape[1]

                jac = np.vstack(
                    [
                        np.sqrt(drho) * jac
                        for drho, jac in zip(
                            drho_df_list, jacobian_list_components_wrt_cur_state
                        )
                    ]
                )
                jac = np.vstack([jac, np.zeros((1, nx))])

                jac_list.append(jac)
            else:
                jac_list.append(None)
        return jac_list
