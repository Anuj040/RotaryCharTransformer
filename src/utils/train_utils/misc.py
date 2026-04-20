import math

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


def get_lr(it: int, config) -> float:
    if it < config["warmup_iters"]:
        return config["learning_rate"] * it / config["warmup_iters"]
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (
        config["lr_decay_iters"] - config["warmup_iters"]
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])
