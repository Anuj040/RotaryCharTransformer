import torch
from torch import nn

from src.utils.model_utilities.model import GPTConfig
from src.utils.model_utilities.model_baseline import BaselineGPT
from src.utils.model_utilities.model_rope import GPTWithRoPE


def select_model(config) -> nn.Module:
    gpt_config_keys = [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
        "dropout",
        "model_type",
    ]
    gpt_config = {k: v for k, v in config.items() if k in gpt_config_keys}
    gptconf = GPTConfig(**gpt_config)

    if config.get("model_type") in ["rope", "nope"]:
        model = GPTWithRoPE(gptconf)
        print("Using GPTWithRoPE model.")
    elif config.get("model_type") == "trm":
        from src.utils.model_utilities.model_rope_trm import TRMGPTWithRoPE

        model = TRMGPTWithRoPE(gptconf)
        print("Using TRMGPTWithRoPE model.")
    else:
        model = BaselineGPT(gptconf)
        print("Using BaselineGPT model.")
    if torch.cuda.is_available():
        model = torch.compile(model)
    return model
