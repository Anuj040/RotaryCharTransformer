import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig
from model_rope import Block, GPTWithRoPE

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
        self.num_recursive_steps = getattr(config, "num_recursive_steps", 6)
        self.num_deep_recursions = getattr(config, "num_deep_recursions", 3)

        # Main transformer container
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))

        # Blocks: either single shared block (for recursion) or the original list
        if self.share_blocks:
            # Tiny net: one Block used recursively
            # For simplicity we always enable RoPE here; 
            # 2 layers seem to be optimal: https://arxiv.org/pdf/2510.04871
            self.transformer["h"] = nn.ModuleList(
                [Block(config, True) for _ in range(2)]
            )
        else:
            # Original behavior: stack of distinct Blocks
            self.transformer["h"] = nn.ModuleList([
                Block(
                    config,
                    (ind + 1) % 4 != 0 if config.model_type == "nope" else True
                )
                for ind in range(config.n_layer)
            ])

        self.transformer["ln_f"] = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # --- TRM-style fixed random initial states --- #
        # register buffers like TRM's H_init / L_init
        self.register_buffer(
            "H_init",
            torch.empty(config.n_embd, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            torch.empty(config.n_embd, dtype=torch.float32),
            persistent=True,
        )
        # truncated normal-ish init
        nn.init.trunc_normal_(self.H_init, mean=0.0, std=1.0)
        nn.init.trunc_normal_(self.L_init, mean=0.0, std=1.0)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_latent_states(self, b: int, t: int, device: torch.device, dtype: torch.dtype):
        """
        Broadcast the fixed random H_init / L_init to [B, T, C].
        This is analogous to TRM's reset_carry + empty_carry, but
        """
        # H_init/L_init: [C]  -> [1, 1, C] -> [B, T, C]
        H = self.H_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(b, t, -1)
        L = self.L_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(b, t, -1)
        return H, L

    # -------- TRM-style pieces: latent recursion & deep recursion -------- #

    def _latent_recursion(self, tok_emb: torch.Tensor, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        """
        latent recursion: apply the shared Block num_recursive_steps times.

        This corresponds to the inner loop in the pseudocode:

            for i in range(n):
                z = net(x, y, z)
            y = net(y, z)
        """

        for n_step in range(self.num_recursive_steps):
            if n_step + 1 == self.num_recursive_steps:
                for block in self.transformer.h:
                    z_L = block(z_L + z_H + tok_emb)
            else:
                for block in self.transformer.h:
                    z_H = block(z_L + z_H)
        return z_H, z_L

    def _deep_recursion(self, tok_emb: torch.Tensor, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
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
                # explicitly detach to avoid any chance of graph accumulation
                z_L = z_L.detach()
                z_H = z_H.detach()

        # Final step: gradients flow
        z_H, _ = self._latent_recursion(tok_emb, z_H, z_L)
        return z_H

    # --------------------------------------------------------------------- #

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        
        # init latent states z_H, z_L from fixed random buffers
        z_H, z_L = self._init_latent_states(b, t, device, tok_emb.dtype)
        if self.share_blocks:
            # TRM-like recursive tiny model with truncated BPTT
            x = self._deep_recursion(x, z_H, z_L)
        else:
            # Original stack of Blocks
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference: only last time step
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss