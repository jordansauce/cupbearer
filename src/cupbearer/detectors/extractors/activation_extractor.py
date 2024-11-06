from typing import Any, Callable

import torch

from cupbearer import utils

from .core import FeatureCache, FeatureExtractor


class ActivationExtractor(FeatureExtractor):
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
        get_grads: bool = False,
        output_func_for_grads: Callable[[torch.Tensor], torch.Tensor] | None = None,
        act_grad_combination_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ):
        """Get activations from a model"""
        # global_processing_fn might return a dict with different keys than the names
        # of raw activations. In that case, the user should pass in processed_names
        # with a list of the names that global_processing_fn returns, whereas names
        # will be the raw activation names.
        # We need to know the processed names to pass them on to the cache.
        # The cache always stores processed instead of raw activations, since in some
        # cases these may be much smaller (e.g. if we only use certain token positions).
        # TODO(erik): maybe we should get these names automatically by running
        # global_processing_fn on an example batch.
        # TODO(erik): it's sad that we can't cache raw activations in cases where this
        # would make more sense.
        super().__init__(
            feature_names=processed_names or names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache,
        )
        self.names = names
        self.get_grads = get_grads
        self.output_func_for_grads = output_func_for_grads
        self.act_grad_combination_func = act_grad_combination_func

        # if self.get_grads:
        #     for name in self.names:
        #         if name.endswith(".input") or name.endswith(".output"):
        #             grad_name = name + "_grad"
        #             if grad_name not in self.names:
        #                 self.names.append(grad_name)

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        if self.get_grads:
            assert (
                self.output_func_for_grads is not None
            ), "output_func_for_grads must be provided when get_grads is True"
            return utils.get_activations_and_grads(
                inputs,
                model=self.model,
                names=self.names,
                output_func_for_grads=self.output_func_for_grads,
                act_grad_combination_func=self.act_grad_combination_func,
            )
        else:
            return utils.get_activations(inputs, model=self.model, names=self.names)
