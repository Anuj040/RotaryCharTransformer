import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.model_utilities.model import GPTConfig
from src.utils.model_utilities.model_rope import Block, GPTWithRoPE


class TRMGPTWithRoPE(GPTWithRoPE):
    """
    GPT with RoPE, extended with TRM-like recursive flow.

    Two modes:

    1. Normal (non-recursive, default):
       - config.share_blocks = False
       - self.transformer.h is a ModuleList of distinct Blocks
       - forward() is exactly as the original code.

    2. Tiny Recursive (TRM-style):
       - config.share_blocks = True
       - self.transformer.h is a *single* shared Block
       - forward() implements:
         latent_recursion(x): apply that Block num_recursive_steps times.
         deep_recursion(x):  run latent_recursion T-1 times under no_grad,
                             then once with grad.

       This matches the pseudocode: T-1 latent reasoning passes without
       gradients, one pass with gradients, per forward.
    """

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # TRM-like flags (fallback to safe defaults if missing)
        self.share_blocks = getattr(config, "share_blocks", True)
        self.num_recursive_steps = getattr(
            config, "num_recursive_steps", 4
        )  # 2) #4)#5)
        self.num_deep_recursions = getattr(config, "num_deep_recursions", 2)  # 2)

        # Main transformer container
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size,
                    config.n_embd // 4 if self.config.perlayerembeds else config.n_embd,
                ),
                drop=nn.Dropout(config.dropout),
            )
        )

        # Blocks: either single shared block (for recursion) or the original list
        if self.share_blocks:
            # Tiny net: one Block used recursively
            # For simplicity we always enable RoPE here;
            # 2 layers seem to be optimal: https://arxiv.org/pdf/2510.04871
            self.transformer["h"] = nn.ModuleList(
                [Block(config, True, cross_attn=False) for _ in range(2)]
            )
            if self.config.perlayerembeds:
                perlayerembeds = [
                    nn.Linear(config.n_embd // 4, config.n_embd, bias=False)
                    for _ in range(3)
                ]
            else:
                perlayerembeds = [nn.Identity() for _ in range(3)]
            self.transformer["proj"] = nn.ModuleList(perlayerembeds)

        else:
            # Original behavior: stack of distinct Blocks
            self.transformer["h"] = nn.ModuleList(
                [
                    Block(
                        config,
                        (ind + 1) % 4 != 0 if config.model_type == "nope" else True,
                    )
                    for ind in range(config.n_layer)
                ]
            )

        self.transformer["ln_f"] = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(
            config.n_embd // 4 if self.config.perlayerembeds else config.n_embd,
            config.vocab_size,
            bias=False,
        )
        # weight tying
        self.lm_head.weight = self.transformer.wte.weight

        self.ln_h = nn.LayerNorm(config.n_embd)
        self.ln_l = nn.LayerNorm(config.n_embd)

        self.a_L = nn.Parameter(torch.tensor(1.0 / 3.0))
        self.a_H = nn.Parameter(torch.tensor(1.0 / 3.0))
        self.a_X = nn.Parameter(torch.tensor(1.0 / 3.0))

        # self.n_L = nn.LayerNorm(config.n_embd)
        # self.n_H = nn.LayerNorm(config.n_embd)
        # self.n_X = nn.LayerNorm(config.n_embd)

        self.b_L = nn.Parameter(torch.tensor(0.5))
        self.b_H = nn.Parameter(torch.tensor(0.5))

        # self.n_L2 = nn.LayerNorm(config.n_embd)
        # self.n_H2 = nn.LayerNorm(config.n_embd)

        # self.alpha_L = nn.Parameter(torch.tensor(0.1))  # start small
        # self.alpha_H = nn.Parameter(torch.tensor(0.1))

        self.q_head = nn.Linear(config.n_embd, 2, bias=True)
        self.down_proj = (
            nn.Linear(config.n_embd, config.n_embd // 4, bias=False)
            if self.config.perlayerembeds
            else nn.Identity()
        )
        # init like TRM: bias to negative so initial halt prob is low
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # -------- TRM-style pieces: latent recursion & deep recursion -------- #

    def _latent_recursion(
        self, tok_emb: torch.Tensor, z_H: torch.Tensor, z_L: torch.Tensor
    ) -> torch.Tensor:
        """
        latent recursion: apply the shared Block num_recursive_steps times.

        This corresponds to the inner loop in the pseudocode:

            for i in range(n):
                z = net(x, y, z)
            y = net(y, z)
        """

        for _ in range(self.num_recursive_steps - 1):
            for ind, block in enumerate(self.transformer.h):
                z_L = self.ln_l(
                    block(
                        self.a_L * z_L
                        + self.a_H * z_H
                        + self.a_X * self.transformer["proj"][ind](tok_emb)
                    )
                )
                # z_L = block(z_L, z_L + z_H + tok_emb)

                # u = self.a_L * self.n_L(z_L) + self.a_H * self.n_H(z_H) + self.a_X * self.n_X(tok_emb)
                # delta_L = block(u)
                # z_L = z_L + self.alpha_L * delta_L
        for block in self.transformer.h:
            z_H = self.ln_h(block(self.b_L * z_L + self.b_H * z_H))
            # z_H = block(z_H, z_L + z_H)
            # v = self.b_L * self.n_L2(z_L) + self.b_H * self.n_H2(z_H)  # separate params/norms are often helpful
            # delta_H = block(v)
            # z_H = z_H + self.alpha_H * delta_H
        return z_H, z_L

    def _deep_recursion(
        self, tok_emb: torch.Tensor, z_H: torch.Tensor, z_L: torch.Tensor
    ) -> torch.Tensor:
        """
        deep recursion: T-1 latent recursions in no_grad, 1 with gradients.

        Pseudocode mapping:

            for j in range(T-1):    # no grad
                with torch.no_grad():
                    y, z = latent_recursion(...)
            # last one with grad
            y, z = latent_recursion(...)
        """
        T = max(int(self.num_deep_recursions), 1)

        # First T-1 steps: no gradients
        for _ in range(T - 1):
            with torch.no_grad():
                z_H, z_L = self._latent_recursion(tok_emb, z_H, z_L)

        # for _ in range(T - 1):
        #     z_H, z_L = self._latent_recursion(tok_emb, z_H, z_L)

        # Final step: gradients flow
        z_H, z_L = self._latent_recursion(tok_emb, z_H, z_L)
        return z_H, z_L

    # --------------------------------------------------------------------- #

    def forward(self, idx, targets=None, z_H=None, z_L=None) -> tuple[torch.Tensor]:
        idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        # init latent states z_H, z_L from fixed random buffers
        if z_H is None or z_L is None:
            x_up = self.transformer["proj"][-1](x)
            z_H = self.ln_h(x_up)
            z_L = self.ln_l(x_up)
            # z_H = x
            # z_L = x
        if self.share_blocks:
            # TRM-like recursive tiny model with truncated BPTT
            z_H, z_L = self._deep_recursion(x, z_H, z_L)
            x = z_H

            # Q-head logits per token: [B, T, 2]
            q_halt_logits = self.q_head(z_H)
        else:
            # Original stack of Blocks
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            q_halt_logits = None  # no q_head in non-recursive mode

        if targets is not None:
            logits = self.lm_head(self.down_proj(x))
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference: only last time step
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, z_H.detach(), z_L.detach(), q_halt_logits
