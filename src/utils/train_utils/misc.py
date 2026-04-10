import torch


def scale_hyperparams(
    current_step: int,
    max_steps: int,
    hyper_param: float,
    scores: torch.Tensor = None,
    min_value: float = None,
) -> float:
    if current_step >= max_steps:
        return hyper_param

    if scores is None:
        scaled_value = hyper_param * ((current_step + 1) / max_steps) ** 1
    else:
        top_p = 0.10
        scaled_value = torch.quantile(scores, 1.0 - top_p).item()

    if min_value is not None:
        scaled_value = max(scaled_value, min_value)
    return scaled_value
