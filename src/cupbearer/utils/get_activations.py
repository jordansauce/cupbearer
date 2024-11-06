from contextlib import nullcontext
from typing import Callable

import torch


class _Finished(Exception):
    pass


def get_activations(
    *args,
    model: torch.nn.Module,
    names: list[str],
    return_output: bool = False,
    no_grad: bool = True,
    **kwargs,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Get the activations of the model for the given inputs.

    Args:
        model: The model to get the activations from.
        names: The names of the modules to get the activations from. Should be a list
            of strings corresponding to pytorch module names, with ".input" or ".output"
            appended to the end of the name to specify whether to get the input
            or output of the module.
        return_output: If True, returns a tuple of activations and the output of
            the model instead of just activations. Defaults to False so that we can
            stop forward passes early if possible.
        no_grad: If True (default), the forward pass is wrapped in a torch.no_grad()
            block. Set to False if you need to use activations in a backward pass.
        *args: Arguments to pass to the model.
        **kwargs: Keyword arguments to pass to the model.

    Returns:
        A dictionary mapping the names of the modules to the activations of the model
        at that module. Keys contain ".input" or ".output" just like `names`.
        If `return_output` is True, returns a tuple of activations and the output of
        the model.
    """
    activations = {}
    hooks = []

    try:
        all_module_names = [name for name, _ in model.named_modules()]

        for name in names:
            assert name.endswith(".input") or name.endswith(
                ".output"
            ), f"Invalid name {name}, names should end with '.input' or '.output'"
            base_name = ".".join(name.split(".")[:-1])
            assert (
                base_name in all_module_names
            ), f"{base_name} is not a submodule of the model"

        def make_hook(name, is_input):
            def hook(module, input, output):
                if is_input:
                    if isinstance(input, torch.Tensor):
                        activations[name] = input
                    elif isinstance(input[0], torch.Tensor):
                        activations[name] = input[0]
                    else:
                        raise ValueError(
                            "Expected input to be a tensor or tuple with tensor as "
                            f"first element, got {type(input)}"
                        )
                else:
                    activations[name] = output

                if set(names).issubset(activations.keys()) and not return_output:
                    # HACK: stop the forward pass to save time
                    raise _Finished()

            return hook

        for name, module in model.named_modules():
            if name + ".input" in names:
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".input", True))
                )
            if name + ".output" in names:
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".output", False))
                )
        with torch.no_grad() if no_grad else nullcontext():
            try:
                output = model(*args, **kwargs)
            except _Finished:
                assert not return_output
    finally:
        # Make sure we always remove hooks even if an exception is raised
        for hook in hooks:
            hook.remove()

    if return_output:
        return activations, output
    else:
        return activations


# def get_activations_and_grads_keywords(
#     *args,
#     model: torch.nn.Module,
#     names: list[str],
#     return_output: bool = False,
#     no_grad: bool = True,
#     output_func_for_grads: Callable[[torch.Tensor], torch.Tensor] | None = None,
#     **kwargs,
# ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
#     """Get the activations and / or gradients of the model for the given inputs.

#     Args:
#         model: The model to get the activation or gradients (aka "features") from.
#         names: The names of the modules to get the activations from. Should be a list
#             of strings corresponding to pytorch module names, with ".input", ".output"
#             ".input_grad" or ".output_grad" appended to the end of the name to specify
#             whether to get the input or output (or gradients) of the module.
#         return_output: If True, returns a tuple of features and the output of
#             the model instead of just features. Defaults to False so that we can
#             stop forward passes early if possible.
#         no_grad: If True (default), the forward pass is wrapped in a torch.no_grad()
#             block. Set to False if you need to use activations in backward pass. This
#             is overridden if any of the names end with ".input_grad" or ".output_grad"
#         output_func_for_grads: A function that takes the output of the model and
#             reduces it to a (batch_size, ) shaped tensor, which gradients are
#             calculated with respect to. Required when any of the names end with
#             ".input_grad" or ".output_grad".
#         *args: Arguments to pass to the model.
#         **kwargs: Keyword arguments to pass to the model.

#     Returns:
#         features: A dictionary mapping the names of the modules to the activations or
#             gradients of the model at that module. Keys contain ".input", ".output",
#             ".input_grad" or ".output_grad" just like `names`.
#         If `return_output` is True, returns a tuple of (features, outputs), where
#         outputs are a tensor of the outputs of the model (eg. the logits).

#     Returns:
#         `(activations, gradients)` where both are dictionaries as in `get_activations`
#     """
#     features = {}
#     hooks = []
#     base_names = []

#     try:
#         all_module_names = [name for name, _ in model.named_modules()]

#         for name in names:
#             suffix = name.split(".")[-1]
#             assert suffix in ["input", "output", "input_grad", "output_grad"],\
#                 f"Invalid name {name}, names should end with '.input', '.output', "\
#                 "'.input_grad' or '.output_grad'"

#             base_name = ".".join(name.split(".")[:-1])
#             assert (
#                 base_name in all_module_names
#             ), f"{base_name} is not a submodule of the model"

#             if base_name not in base_names:
#                 base_names.append(base_name)

#             if "grad" in suffix:
#                 no_grad = False

#         if output_func_for_grads is None and not no_grad:
#             raise ValueError("output_func_for_grads is required for gradients")

#         def make_forward_hook(name, is_input):
#             def forward_hook(module, input, output):
#                 if is_input:
#                     if isinstance(input, torch.Tensor):
#                         features[name] = input
#                     elif isinstance(input[0], torch.Tensor):
#                         features[name] = input[0]
#                     else:
#                         raise ValueError(
#                             "Expected input to be a tensor or tuple with tensor as "
#                             f"first element, got {type(input)}"
#                         )
#                 if set(names).issubset(features.keys()) and no_grad:
#                     if not return_output:
#                         # HACK: stop the forward pass to save time
#                         raise _Finished()
#             return forward_hook

#         def make_backward_hook(name, is_input):
#             def backward_hook(module, grad_input, grad_output):
#                 if isinstance(grad_input, tuple):
#                     grad_input, *_ = grad_input
#                 if isinstance(grad_output, tuple):
#                     grad_output, *_ = grad_output

#                 if is_input:
#                     features[name] = grad_input
#                 else:
#                     features[name] = grad_output

#                 if set(names).issubset(features.keys()):
#                     # HACK: stop the backward pass to save time
#                     raise _Finished()
#             return backward_hook

#         for name, module in model.named_modules():
#             if name + ".input" in names:
#                 hooks.append(module.register_forward_hook(make_forward_hook(
#                   name + ".input", True)))
#             if name + ".output" in names:
#                 hooks.append(module.register_forward_hook(make_forward_hook(
#                   name + ".output", False)))
#             if name + ".input_grad" in names:
#                 hooks.append(module.register_full_backward_hook(make_backward_hook(
#                   name + ".input_grad", True)))
#             if name + ".output_grad" in names:
#                 hooks.append(module.register_full_backward_hook(make_backward_hook(
#                   name + ".output_grad", False)))
#         with torch.no_grad() if no_grad else torch.enable_grad():
#             try:
#                 output = model(*args, **kwargs)
#                 if not no_grad:
#                     if output_func_for_grads is None:
#                         loss = output[...,-1,:].sum(dim=-1)
#                     else:
#                         loss = output_func_for_grads(output)
#                     assert loss.ndim == 1, "output_func_for_grads should"\
#                           "reduce to a 1D tensor"
#                     loss.backward(torch.ones_like(loss))
#             except _Finished:
#                 pass
#     finally:
#         # Make sure we always remove hooks even if an exception is raised
#         for hook in hooks:
#             hook.remove()
#         if not no_grad:
#             model.zero_grad()

#     if return_output:
#         return features, output
#     else:
#         return features


def get_activations_and_grads(
    *args,
    model: torch.nn.Module,
    names: list[str],
    output_func_for_grads: Callable[[torch.Tensor], torch.Tensor],
    act_grad_combination_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Get the activations and gradients of the model for the given inputs.

    Args:
        See `get_activations`.
        output_func_for_grads: A function that takes the output of the model and reduces
            to a (batch_size, ) shaped tensor.

        act_grad_combination_func(acts, grads): A function that takes activations and
            gradients (both shaped (batch, hidden)) and returns a tensor of shape
            (batch, anything). if None, this will be torch.cat([acts, grads], dim=-1)

    Returns:
        `(activations, gradients)` where both are dictionaries as in `get_activations`.
    """
    activations = {}
    gradients = {}
    hooks = []

    try:
        all_module_names = [name for name, _ in model.named_modules()]

        for name in names:
            assert name.endswith(".input") or name.endswith(
                ".output"
            ), f"Invalid name {name}, names should end with '.input' or '.output'"
            base_name = ".".join(name.split(".")[:-1])
            assert (
                base_name in all_module_names
            ), f"{base_name} is not a submodule of the model"

        def make_hooks(name, is_input):
            def forward_hook(module, input, output):
                if is_input:
                    if isinstance(input, torch.Tensor):
                        activations[name] = input
                    elif isinstance(input[0], torch.Tensor):
                        activations[name] = input[0]
                    else:
                        raise ValueError(
                            "Expected input to be a tensor or tuple with tensor as "
                            f"first element, got {type(input)}"
                        )
                else:
                    activations[name] = output

            def backward_hook(module, grad_input, grad_output):
                if isinstance(grad_input, tuple):
                    grad_input, *_ = grad_input
                if isinstance(grad_output, tuple):
                    grad_output, *_ = grad_output

                if is_input:
                    gradients[name] = grad_input
                else:
                    gradients[name] = grad_output

                if set(names).issubset(gradients.keys()):
                    # HACK: stop the backward pass to save time
                    raise _Finished()

            return forward_hook, backward_hook

        for name, module in model.named_modules():
            if name + ".input" in names:
                forward_hook, backward_hook = make_hooks(name + ".input", True)
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))
            if name + ".output" in names:
                forward_hook, backward_hook = make_hooks(name + ".output", False)
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))
        with torch.enable_grad():
            try:
                out = model(*args, **kwargs)
                out = output_func_for_grads(out)
                assert (
                    out.ndim == 1
                ), "output_func_for_grads should reduce to a 1D tensor"
                out.backward(torch.ones_like(out))
            except _Finished:
                pass
    finally:
        # Make sure we always remove hooks even if an exception is raised
        for hook in hooks:
            hook.remove()

        model.zero_grad()

    # Combine activations and gradients
    for name in activations.keys():
        if act_grad_combination_func is None:
            print(
                "act_grad_combination_func is None: "
                "concatenating activations and gradients"
            )
            activations[name] = torch.cat([activations[name], gradients[name]], dim=-1)
        else:
            activations[name] = act_grad_combination_func(
                activations[name], gradients[name]
            )
        activations[name] = activations[name].detach()
    return activations
