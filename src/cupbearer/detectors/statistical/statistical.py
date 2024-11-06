from abc import abstractmethod
from typing import Any, Callable

import torch
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from cupbearer.detectors.activation_based import ActivationBasedDetector
from cupbearer.detectors.statistical.helpers import update_covariance


class StatisticalDetector(ActivationBasedDetector):
    use_trusted: bool = True
    use_untrusted: bool = False

    @abstractmethod
    def init_variables(
        self, activation_sizes: dict[str, torch.Size], device, case: str
    ):
        pass

    @abstractmethod
    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        pass

    def _train(
        self,
        trusted_dataloader,
        untrusted_dataloader,
        *,
        pbar: bool = True,
        max_steps: int | None = None,
        **kwargs,
    ):
        all_dataloaders = {}
        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        with torch.no_grad():
            if self.use_trusted:
                if trusted_dataloader is None:
                    raise ValueError(
                        f"{self.__class__.__name__} requires trusted training data."
                    )
                all_dataloaders["trusted"] = trusted_dataloader
            if self.use_untrusted:
                if untrusted_dataloader is None:
                    raise ValueError(
                        f"{self.__class__.__name__} requires untrusted training data."
                    )
                all_dataloaders["untrusted"] = untrusted_dataloader

            for case, dataloader in all_dataloaders.items():
                logger.debug(f"Collecting statistics on {case} data")
                _, example_activations = next(iter(dataloader))

                # v is an entire batch, v[0] are activations for a single input
                activation_sizes = {
                    k: v[0].size() for k, v in example_activations.items()
                }
                self.init_variables(
                    activation_sizes,
                    device=next(iter(example_activations.values())).device,
                    case=case,
                )

                if pbar:
                    dataloader = tqdm(dataloader, total=max_steps or len(dataloader))

                for i, (_, activations) in enumerate(dataloader):
                    if max_steps and i >= max_steps:
                        break
                    self.batch_update(activations, case)


class ActivationCovarianceBasedDetector(StatisticalDetector):
    """Generic abstract detector that learns means and covariance matrices
    during training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._means: dict[
            str, dict[str, torch.Tensor]
        ] = {}  # Means {case ('trusted' or 'untrusted): {name: (batch, dim) } }
        self._Cs: dict[
            str, dict[str, torch.Tensor]
        ] = {}  # Covariance matrices (batch, dim, dim)
        self._ns: dict[str, dict[str, int]] = {}  # number of samples

    def init_variables(
        self, activation_sizes: dict[str, torch.Size], device, case: str
    ):
        if any(len(size) != 1 for size in activation_sizes.values()):
            logger.debug(
                "Received multi-dimensional activations, will only learn "
                "covariances along last dimension and treat others independently. "
                "If this is unintentional, pass "
                "`activation_preprocessing_func=utils.flatten_last`."
            )
        logger.debug(
            "Activation sizes: \n"
            + "\n".join(f"{k}: {size}" for k, size in activation_sizes.items())
        )
        self._means[case] = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self._Cs[case] = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._ns[case] = {k: 0 for k in activation_sizes.keys()}

    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            activation = rearrange(activation, "batch ... dim -> (batch ...) dim")
            assert activation.ndim == 2, activation.shape
            (
                self._means[case][k],
                self._Cs[case][k],
                self._ns[case][k],
            ) = update_covariance(
                self._means[case][k],
                self._Cs[case][k],
                self._ns[case][k],
                activation,
            )

    @abstractmethod
    def post_covariance_training(self, **kwargs):
        pass

    @abstractmethod
    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        """Compute the anomaly score for a single layer.

        `name` is passed in to access the mean/covariance and any custom derived
        quantities computed in post_covariance_training.

        `activation` will always have shape (batch, dim). The `batch` dimension might
        not just be the actual batch dimension, but could also contain multiple entries
        from a single sample, in the case of multi-dimensional activations that we
        treat as independent along all but the last dimension.

        Should return a tensor of shape (batch,) with the anomaly scores.
        """
        pass

    def _compute_layerwise_scores(self, inputs, features):
        batch_size = next(iter(features.values())).shape[0]
        features = {
            k: rearrange(v, "batch ... dim -> (batch ...) dim")
            for k, v in features.items()
        }
        scores = {
            k: self._individual_layerwise_score(k, v) for k, v in features.items()
        }
        scores = {
            k: rearrange(
                v,
                "(batch independent_dims) -> batch independent_dims",
                batch=batch_size,
            ).mean(-1)
            for k, v in scores.items()
        }
        return scores

    def _train(self, trusted_dataloader, untrusted_dataloader, **kwargs):
        super()._train(
            trusted_dataloader=trusted_dataloader,
            untrusted_dataloader=untrusted_dataloader,
            **kwargs,
        )

        # Post process
        with torch.inference_mode():
            self.means = self._means
            self.covariances = {}
            for case, Cs in self._Cs.items():
                self.covariances[case] = {
                    k: C / (self._ns[case][k] - 1) for k, C in Cs.items()
                }
                if any(
                    torch.count_nonzero(C) == 0 for C in self.covariances[case].values()
                ):
                    raise RuntimeError("All zero covariance matrix detected.")

            self.post_covariance_training(**kwargs)


class BasisInvariantAttributionDetector(ActivationCovarianceBasedDetector):
    def __init__(
        self,
        n_svals=10,
        feature_extractor=None,
        activation_names: list[str] | None = None,
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
        processed_names: list[str] | None = None,
        cache=None,
        output_func_for_grads: Callable[[torch.Tensor], torch.Tensor] | None = None,
        act_grad_combination_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        **kwargs,
    ):
        if act_grad_combination_func is None:

            def stack_activations_and_grads(activations, grads):
                return torch.stack((activations[..., -1, :], grads[..., -1, :]), dim=-1)

            act_grad_combination_func = stack_activations_and_grads

        if feature_extractor is None:
            if activation_names is None:
                raise ValueError(
                    "Either a feature extractor or a list of activation names "
                    "must be provided."
                )
            from cupbearer.detectors.extractors import GradientFeatureExtractor

            feature_extractor = GradientFeatureExtractor(
                names=activation_names,
                individual_processing_fn=individual_processing_fn,
                global_processing_fn=global_processing_fn,
                processed_names=processed_names,
                cache=cache,
                output_func_for_grads=output_func_for_grads,
                act_grad_combination_func=act_grad_combination_func,
            )

        super().__init__(
            feature_extractor=feature_extractor,
            activation_names=activation_names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            processed_names=processed_names,
            cache=cache,
            output_func_for_grads=output_func_for_grads,
            act_grad_combination_func=act_grad_combination_func,
            **kwargs,
        )

        self.n_svals = n_svals  # Number of singular vectors to keep
        self._act_covs: dict[
            str, dict[str, torch.Tensor]
        ] = {}  # Covariance matrices (batch, dim, dim)
        self._grad_covs_uncentered: dict[
            str, dict[str, torch.Tensor]
        ] = {}  # Uncentered gradient covariance matrices (batch, dim, dim)
        self.projector_half_A: dict[str, dict[str, torch.Tensor]] = {}
        self.projector_half_G: dict[str, dict[str, torch.Tensor]] = {}

    def init_variables(
        self, activation_sizes: dict[str, torch.Size], device, case: str
    ):
        act_sizes = {k: v[:-1] for k, v in activation_sizes.items()}
        grad_sizes = {k: v[:-1] for k, v in activation_sizes.items()}
        super().init_variables(act_sizes, device, case)
        self._grad_covs_uncentered[case] = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in grad_sizes.items()
        }

    def update_grad_covs(self, grads: dict[str, torch.Tensor], case: str):
        for k, grad in grads.items():
            grad = rearrange(grad, "batch ... dim -> (batch ...) dim")
            assert grad.ndim == 2, grad.shape
            prev_factor = (self._ns[case][k] - grad.shape[0]) / self._ns[case][k]
            new_factor = grad.shape[0] / self._ns[case][k]
            self._grad_covs_uncentered[case][
                k
            ] = prev_factor * self._grad_covs_uncentered[case][
                k
            ] + new_factor * torch.einsum(
                "bi,bj->ij", grad, grad
            )

    def update_projection_matrices(self, case: str):
        assert self._Cs[case].keys() == self._grad_covs_uncentered[case].keys()
        self.projector_half_A[case] = {}
        self.projector_half_G[case] = {}
        for k in self._Cs[case].keys():
            act_cov = self._Cs[case][k] / (self._ns[case][k] - 1)

            # Update projection matrices
            A_vecs, A_vals, _ = torch.linalg.svd(act_cov, full_matrices=False)
            G_vecs, G_vals, _ = torch.linalg.svd(
                self._grad_covs_uncentered[case][k], full_matrices=False
            )
            A = A_vecs * A_vals.unsqueeze(-2) ** 0.5
            G = G_vecs * G_vals.unsqueeze(-2) ** 0.5
            U, S, Vh = torch.linalg.svd(G.mT @ A)
            U = U[..., : self.n_svals]
            S = S[..., : self.n_svals].unsqueeze(-2)
            # S /= S.max(dim=-1, keepdim=True).values
            S = S ** (-0.5)
            V = Vh.mT[..., : self.n_svals]
            self.projector_half_A[case][k] = A @ (V * S)  # default is S ** (-0.5)
            self.projector_half_G[case][k] = G @ (U * S)  # default is S ** (-0.5)
            # -0.5: 88.0, -2.0: 85.0, -1.0: 80.3, -3.0: 85.9, -0.6: 71.9, -0.4: 69.79

    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        acts = {k: v[..., 0] for k, v in activations.items()}
        grads = {k: v[..., 1] for k, v in activations.items()}
        super().batch_update(acts, case)
        self.update_grad_covs(grads, case)

    def _compute_layerwise_scores(self, inputs, features):
        batch_size = next(iter(features.values())).shape[0]
        features = {
            k: rearrange(v, "batch ... dim isgrad -> (batch ...) dim isgrad")
            for k, v in features.items()
        }
        scores = {
            k: self._individual_layerwise_score(k, v) for k, v in features.items()
        }
        scores = {
            k: rearrange(
                v,
                "(batch independent_dims) -> batch independent_dims",
                batch=batch_size,
            ).mean(-1)
            for k, v in scores.items()
        }
        return scores

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        act = activation[..., 0]
        grad = activation[..., 1]

        act_centered = act - self.means["trusted"][name]
        score_I = torch.einsum("bd,bd->b", grad, act_centered)
        grad_transformed = (
            grad @ self.projector_half_A["trusted"][name]
        )  # (batch, dim), (dim, k) -> (batch, k)
        act_transformed = (
            act_centered @ self.projector_half_G["trusted"][name]
        )  # (batch, dim), (dim, k) -> (batch, k)
        score_P = torch.einsum("bd,bd->b", grad_transformed, act_transformed)

        score = (score_I - score_P) ** 2

        print("")
        print(
            "act_centered norm = "
            f"{torch.linalg.norm(act_centered, ord=2, dim=-1).mean()}"
        )
        print(
            "act_transformed norm = "
            f"{torch.linalg.norm(act_transformed, ord=2, dim=-1).mean()}"
        )
        print(f"grad norm = {torch.linalg.norm(grad, ord=2, dim=-1).mean()}")
        print(
            f"grad_transformed norm = "
            f"{torch.linalg.norm(grad_transformed, ord=2, dim=-1).mean()}"
        )
        print(f"mean score_I = {score_I.mean()}")
        print(f"mean score_P = {score_P.mean()}")
        print(f"mean score = {score.mean()}")

        return score

    def _get_trained_variables(self):
        return {
            "means": self.means,
            "projector_half_A": self.projector_half_A,
            "projector_half_G": self.projector_half_G,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.projector_half_A = variables["projector_half_A"]
        self.projector_half_G = variables["projector_half_G"]

    def post_covariance_training(self, **kwargs):
        pass

    def _train(self, trusted_dataloader, untrusted_dataloader, **kwargs):
        super()._train(
            trusted_dataloader=trusted_dataloader,
            untrusted_dataloader=untrusted_dataloader,
            **kwargs,
        )

        # Post process
        with torch.inference_mode():
            self.means = self._means
            self.update_projection_matrices("trusted")
            self.covariances = {}
            for case, Cs in self._Cs.items():
                self.covariances[case] = {
                    k: C / (self._ns[case][k] - 1) for k, C in Cs.items()
                }
                if any(
                    torch.count_nonzero(C) == 0 for C in self.covariances[case].values()
                ):
                    raise RuntimeError("All zero covariance matrix detected.")
            self.post_covariance_training(**kwargs)
