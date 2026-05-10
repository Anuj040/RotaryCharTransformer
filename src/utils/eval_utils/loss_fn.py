import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.model_utilities.model_rope_trm import TRMGPTWithRoPE


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device,
    config,
    ctx=nullcontext(),
):
    model.eval()
    if isinstance(model, TRMGPTWithRoPE):
        N_supervision = config.get("N_supervised_steps_eval", 2)
    else:
        N_supervision = 1
    losses = torch.zeros((N_supervision))
    for _, (X, Y) in enumerate(val_loader):
        X, Y = X.to(device), Y.to(device)
        z_H, z_L = None, None
        for super_ind in range(N_supervision):
            with ctx:
                if isinstance(model, TRMGPTWithRoPE):
                    _, loss, z_H, z_L, _ = model(X, Y, z_H, z_L)
                else:
                    _, loss = model(X, Y)
            losses[super_ind] += loss.item()
    out = {
        f"val_{super_ind}": losses[super_ind].item() / len(val_loader)
        for super_ind in range(N_supervision)
    }
    model.train()
    return out


@torch.inference_mode()
def estimate_test_loss(
    model: nn.Module,
    val_loader: DataLoader,
    N_supervised_steps: int,
    device,
    ctx=nullcontext(),
):
    losses = np.zeros(N_supervised_steps)
    for X, Y in val_loader:
        X, Y = X.to(device), Y.to(device)
        z_H, z_L = None, None
        for step in range(N_supervised_steps):
            with ctx:
                _, loss, z_H, z_L, _ = model(X, Y, z_H, z_L)
            losses[step] += loss.item()
    mean_loss = losses / len(val_loader)
    bpc = mean_loss / math.log(2)
    for step in range(N_supervised_steps):
        print(f"Steps {step}: {mean_loss[step]:.4f} | bpc: {bpc[step]:8.3f}")
