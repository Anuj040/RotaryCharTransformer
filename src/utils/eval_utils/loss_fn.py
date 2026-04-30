import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device,
    config,
    ctx=nullcontext(),
):
    model.eval()
    N_supervision = config.get("N_supervised_steps_eval", 2)
    for split in ["val"]:
        losses = torch.zeros((N_supervision))
        with ctx:
            for _, (X, Y) in enumerate(val_loader):
                X, Y = X.to(device), Y.to(device)
                z_H, z_L = None, None
                for super_ind in range(N_supervision):
                    with ctx:
                        _, loss, z_H, z_L, _ = model(X, Y, z_H, z_L)
                    losses[super_ind] += loss.item()
        out = {
            f"{split}_{super_ind}": losses[super_ind].item() / len(val_loader)
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
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for X, Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            z_H, z_L = None, None
            for step in range(N_supervised_steps):
                with ctx:
                    logits, loss, z_H, z_L, _ = model(X, Y, z_H, z_L)
                losses[step] += loss.item()
    mean_loss = losses / len(val_loader)
    bpc = mean_loss / math.log(2)
    for step in range(N_supervised_steps):
        print(f"Steps {step}: {mean_loss[step]:.4f} | bpc: {bpc[step]:8.3f}")
