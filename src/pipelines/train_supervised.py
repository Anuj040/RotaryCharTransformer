# modified train.py
import argparse
import math
import os
import pickle
import time
from contextlib import nullcontext
from itertools import cycle

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from tqdm import tqdm

import wandb
from src.utils.data_utils.prepare_dataset import EnwikDataset
from src.utils.model_utilities.pick_model import select_model
from src.utils.train_utils.misc import scale_hyperparams


def get_serializable_config(config):
    return {
        k: v
        for k, v in config.items()
        if isinstance(v, (int, float, str, bool, type(None))) and not k.startswith("__")
    }


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device,
    config,
    ctx=nullcontext(),
):
    out = {}
    model.eval()
    for split in ["val"]:
        losses = torch.zeros(len(val_loader))
        with ctx:
            for ind, (X, Y) in enumerate(val_loader):
                X, Y = X.to(device), Y.to(device)
                z_H, z_L = None, None
                for _ in range(config.get("N_supervised_steps_eval", 2)):
                    with ctx:
                        _, loss, z_H, z_L, _ = model(X, Y, z_H, z_L)
                losses[ind] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, config):
    if it < config["warmup_iters"]:
        return config["learning_rate"] * it / config["warmup_iters"]
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (
        config["lr_decay_iters"] - config["warmup_iters"]
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    args = parser.parse_args()

    config_file = args.config
    config = {}
    with open(config_file, "r") as f:
        exec(f.read(), {}, config)

    config = {k: v for k, v in config.items() if not k.startswith("__")}
    assert (
        0 < config["gradient_accumulation_steps"] <= 1
    ), "gradient_accumulation_steps > 1 is not supported in supervised training."
    if "out_dir" not in config:
        print("Error: 'out_dir' not specified in the configuration file.")
        return

    if int(os.environ.get("RANK", -1)) == -1:
        os.makedirs(config["out_dir"], exist_ok=True)
        print(f"Output directory: {config['out_dir']}")

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        config["gradient_accumulation_steps"] //= ddp_world_size
    else:
        master_process = True
        ddp_world_size = 1
        device = config["device"]

    tokens_per_iter = (
        config["gradient_accumulation_steps"]
        * ddp_world_size
        * config["batch_size"]
        * config["block_size"]
    )
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    data_dir = os.path.join("data", config["dataset"])
    train_dataset = EnwikDataset(
        os.path.join(data_dir, "train.bin"), config["block_size"]
    )
    val_dataset = EnwikDataset(os.path.join(data_dir, "val.bin"), config["block_size"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        prefetch_factor=4,
        pin_memory=(device_type == "cuda"),
        num_workers=4 if device_type == "cuda" else 2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2 * config["batch_size"],
        shuffle=False,
        prefetch_factor=8,
        pin_memory=(device_type == "cuda"),
        num_workers=4 if device_type == "cuda" else 2,
    )
    train_loader = cycle(train_loader)

    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    config["vocab_size"] = vocab_size

    model = select_model(config)
    model.to(device)

    # Initialize optimizer outside of the model
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))

    iter_num = 0
    best_val_loss = 1e9
    N_supervised_steps = config.get("N_supervised_steps", 3)

    if config.get("init_from", "scratch") == "resume":
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from iteration {iter_num}, best val loss {best_val_loss}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params/1e6:.2f}M")

    if master_process:
        wandb.init(
            project=config.get("wandb_project", "bpc"),
            name=config.get("wandb_run_name", None),
            config=get_serializable_config(config),
        )

    t0 = time.time()

    local_iter_num = 0
    raw_model = model.module if ddp else model

    with tqdm(total=config["max_iters"]) as pbar:
        train_losses = []
        iter_num = 0
        scores = None
        for X, Y in train_loader:
            z_H, z_L, all_z_H, all_z_L = None, None, None, None
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            B = X.shape[0]
            active_mask = torch.ones(B, dtype=torch.bool, device=device)
            correctness_threshold = scale_hyperparams(
                iter_num,
                config["max_iters"],
                config.get("correctness_threshold", 0.8),
                scores=scores,
                min_value=0.3,
            )
            for _ in range(N_supervised_steps):
                if not active_mask.any():
                    break
                # indices of currently active sequences
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                # slice X/Y and z_H/z_L for active subset
                X_active = X[active_idx]
                Y_active = Y[active_idx]
                if all_z_H is None and all_z_L is None:
                    z_H_active, z_L_active = None, None
                else:
                    z_H_active = all_z_H[active_idx]
                    z_L_active = all_z_L[active_idx]

                lr = (
                    get_lr(iter_num, config)
                    if config["decay_lr"]
                    else config["learning_rate"]
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == config["gradient_accumulation_steps"] - 1
                    )
                with ctx:
                    logits, loss, z_H_new, z_L_new, q_halt_logits = model(
                        X_active, Y_active, z_H_active, z_L_active
                    )
                    if all_z_H is None:
                        all_z_H = torch.empty(
                            (B, *z_H_new.shape[1:]),
                            device=z_H_new.device,
                            dtype=z_H_new.dtype,
                        )
                        all_z_L = torch.empty(
                            (B, *z_L_new.shape[1:]),
                            device=z_L_new.device,
                            dtype=z_L_new.dtype,
                        )
                    all_z_H[active_mask] = z_H_new
                    all_z_L[active_mask] = z_L_new

                    # q_loss, halt_now_active, scores = compute_trm_losses_and_halt(
                    #         logits, Y_active, q_halt_logits,
                    #         correctness_threshold=correctness_threshold,
                    #     )
                    # loss = (loss + q_loss)/ config['gradient_accumulation_steps']
                    # active_mask[active_idx] = active_mask[active_idx] & (~halt_now_active)
                scaler.scale(loss).backward()

                if config["grad_clip"] != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["grad_clip"]
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item() * config["gradient_accumulation_steps"])
            t1 = time.time()
            t1 - t0
            t0 = t1

            if (iter_num + 1) % config["eval_interval"] == 0 and master_process:
                losses = estimate_loss(model, val_loader, device, config, ctx)
                print(
                    f"\nStep {iter_num}: train loss {np.mean(train_losses):.4f}, val loss {losses['val']:.4f} | val_bpc {losses['val'] / math.log(2):8.3f}"
                )
                wandb.log(
                    {
                        "step": iter_num,
                        "train_loss": np.mean(train_losses),
                        "val_loss": losses["val"],
                        "val_bpc": losses["val"] / math.log(2),
                    },
                    step=iter_num,
                )
                train_losses = []
                if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
                    best_val_loss = losses["val"]
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": get_serializable_config(config),
                    }
                    checkpoint_path = os.path.join(config["out_dir"], "ckpt.pt")
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

            iter_num += 1
            local_iter_num += 1
            pbar.update(1)

            if iter_num % config["log_interval"] == 0 and master_process:
                lossf = loss.item() * config["gradient_accumulation_steps"]
                print(
                    f"Iter {iter_num}: loss {lossf:5.2f} | ppl {math.exp(lossf):8.2f} | bpc {lossf / math.log(2):8.3f}"
                )
            if iter_num > config["max_iters"]:
                break

    wandb.finish()
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
