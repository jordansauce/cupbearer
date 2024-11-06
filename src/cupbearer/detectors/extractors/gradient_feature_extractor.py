from typing import Any, Callable

import torch

from cupbearer import utils
from cupbearer.detectors.statistical.helpers import update_covariance

from .core import FeatureCache, FeatureExtractor


class GradientFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        names: list[str],
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
        processed_names: list[str] | None = None,
        cache: FeatureCache | None = None,
        output_func_for_grads: Callable[[torch.Tensor], torch.Tensor] | None = None,
        act_grad_combination_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ):
        """Get activations and gradients from a model, and combine them into features.

        Args:
            output_func_for_grads: The output logits of the model are passed through
                this function before gradients are calculated. Gradients of activations
                will be computed with respect to the output of this function. Should
                return a tensor of shape (batch).
            act_grad_combination_func: A function mapping activations and gradients
                (passed as the first two input args) to a single tensor of features.
        """

        super().__init__(
            feature_names=processed_names or names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache,
        )
        self.names = names
        self.output_func_for_grads = output_func_for_grads
        self.act_grad_combination_func = act_grad_combination_func
        if output_func_for_grads is None:
            # Default output function: sum of the logits at the final token position
            self.output_func_for_grads = lambda logits: torch.sum(
                logits[:, -1, :], dim=-1
            )
        else:
            self.output_func_for_grads = output_func_for_grads

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        return utils.get_activations_and_grads(
            inputs,
            model=self.model,
            names=self.names,
            output_func_for_grads=self.output_func_for_grads,
            act_grad_combination_func=self.act_grad_combination_func,
        )


class BasisInvariantAttributionFeatureExtractor(GradientFeatureExtractor):
    def __init__(self, *args, k=10, **kwargs):
        def stack_activations_and_grads(activations, grads):
            return torch.stack((activations, grads), dim=-1)

        kwargs.update(act_grad_combination_func=stack_activations_and_grads)
        super().__init__(*args, **kwargs)
        self.k = k  # Number of singular vectors to keep
        self._means: dict[str, torch.Tensor] = {}  # (batch, dim)
        self._ns: dict[str, int] = {}  # number of samples
        self._Cs: dict[
            str, torch.Tensor
        ] = {}  # Covariance matrices not yet divided by n (batch, dim, dim)
        self._act_covs: dict[
            str, torch.Tensor
        ] = {}  # Covariance matrices (batch, dim, dim)
        self._grad_covs_uncentered: dict[
            str, torch.Tensor
        ] = {}  # Uncentered gradient covariance matrices (batch, dim, dim)
        self._projector_half_A: dict[str, torch.Tensor] = {}
        self._projector_half_G: dict[str, torch.Tensor] = {}

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        acts_grads = utils.get_activations_and_grads(
            inputs,
            model=self.model,
            names=self.names,
            output_func_for_grads=self.output_func_for_grads,
            act_grad_combination_func=self.act_grad_combination_func,
        )

        # Only keep the last token
        acts_grads = {
            name: acts_grads[name][..., -1, :, :].detach() for name in acts_grads
        }

        with torch.no_grad():
            if not self.frozen:
                for name, act_grad in acts_grads.items():
                    # Update means and covariances
                    acts = act_grad[..., 0]
                    grads = act_grad[..., 1]
                    acts = acts.reshape(-1, acts.shape[-1])
                    grads = grads.reshape(-1, grads.shape[-1])

                    if name not in self._means:
                        self._means[name] = torch.zeros(
                            acts.shape[-1], device=acts.device
                        )
                        self._ns[name] = 0
                        self._Cs[name] = torch.zeros(
                            acts.shape[-1], acts.shape[-1], device=acts.device
                        )
                        self._grad_covs_uncentered[name] = torch.zeros(
                            grads.shape[-1], grads.shape[-1], device=grads.device
                        )

                    (
                        self._means[name],
                        self._Cs[name],
                        self._ns[name],
                    ) = update_covariance(
                        self._means[name], self._Cs[name], self._ns[name], acts
                    )
                    self._act_covs[name] = self._Cs[name] / (self._ns[name] - 1)

                    prev_factor = (self._ns[name] - acts.shape[0]) / self._ns[name]
                    new_factor = acts.shape[0] / self._ns[name]
                    self._grad_covs_uncentered[
                        name
                    ] = prev_factor * self._grad_covs_uncentered[
                        name
                    ] + new_factor * torch.einsum(
                        "bi,bj->ij", grads, grads
                    )

                    # Update projection matrices
                    A_vecs, A_vals, _ = torch.linalg.svd(
                        self._act_covs[name], full_matrices=False
                    )
                    G_vecs, G_vals, _ = torch.linalg.svd(
                        self._grad_covs_uncentered[name], full_matrices=False
                    )
                    A = A_vecs * A_vals.unsqueeze(-2) ** 0.5
                    G = G_vecs * G_vals.unsqueeze(-2) ** 0.5
                    U, S, Vh = torch.linalg.svd(G.mT @ A)
                    U = U[..., : self.k]
                    S = S[..., : self.k].unsqueeze(-2)
                    V = Vh.mT[..., : self.k]
                    self._projector_half_A[name] = A @ (V * S ** (-0.5))
                    self._projector_half_G[name] = G @ (U * S ** (-0.5))

            # Compute features
            relevant_acts = {}
            for name, act_grad in acts_grads.items():
                acts = act_grad[..., 0]
                grads = act_grad[..., 1]
                acts = acts.reshape(-1, acts.shape[-1])
                grads = grads.reshape(-1, grads.shape[-1])

                relevant_acts[name] = acts @ self._projector_half_A[name]  # (batch, k)
                relevant_acts[name] = (
                    relevant_acts[name] @ self._projector_half_G[name].mT
                )
        return relevant_acts
