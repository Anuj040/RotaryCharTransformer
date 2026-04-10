import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
# import wandb

from model import GPTConfig
from model_baseline import BaselineGPT
from src.utils.model_utilities.model_rope import GPTWithRoPE
from itertools import cycle


from src.utils.data_utils.prepare_dataset import EnwikDataset
from src.pipelines.train_supervised import get_serializable_config, estimate_loss, get_lr
from src.utils.halting_loss import compute_trm_losses_and_halt

config_file = "config/enwik8_char_rope_trm.py"
# config_file = "config/enwik8_char_rope_baselineV2.py"
config = {}
with open(config_file, 'r') as f:
    exec(f.read(), {}, config)

config = {k: v for k, v in config.items() if not k.startswith('__')}
assert 0 < config['gradient_accumulation_steps'] <= 1, "gradient_accumulation_steps > 1 is not supported in supervised training."

model_path_temp = ".dbfs/tmp/enwik8"
config["out_dir"] += "_nohlt9V3_2trns4rec_sclnolrnrm2.0"
config["out_dir"] = os.path.join(model_path_temp, config['out_dir'])
os.makedirs(config['out_dir'], exist_ok=True)
ddp = False

master_process = True
ddp_world_size = 1
device = config['device']

tokens_per_iter = (config['gradient_accumulation_steps'] * ddp_world_size *
                    config['batch_size'] * config['block_size'])
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', config['dataset'])
train_dataset = EnwikDataset(os.path.join(data_dir, 'train.bin'), config['block_size'])
val_dataset = EnwikDataset(os.path.join(data_dir, 'val.bin'), config['block_size'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True,  prefetch_factor=4, pin_memory=(device_type=='cuda'), num_workers=8 if device_type=='cuda' else 2
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=3 * config['batch_size'], shuffle=False,  prefetch_factor=8, pin_memory=(device_type=='cuda'), num_workers=8 if device_type=='cuda' else 2
)
train_loader = cycle(train_loader)

meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
config['vocab_size'] = vocab_size

gpt_config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout', 'model_type']
gpt_config = {k: v for k, v in config.items() if k in gpt_config_keys}
gptconf = GPTConfig(**gpt_config)

if config.get('model_type') in ['rope', 'nope']:
    model = GPTWithRoPE(gptconf)
    print("Using GPTWithRoPE model.")
elif config.get('model_type') == 'trm':
    from utils.model_utilities.model_rope_trm import TRMGPTWithRoPE
    model = TRMGPTWithRoPE(gptconf)
    print("Using TRMGPTWithRoPE model.")
else:
    model = BaselineGPT(gptconf)
    print("Using BaselineGPT model.")

model.to(device)

# Initialize optimizer outside of the model
decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': config['weight_decay']},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))

iter_num = 0
best_val_loss = 1e9
N_supervised_steps = config.get('N_supervised_steps', 3)

if config.get('init_from', 'scratch') == 'resume':
    print(f"Resuming training from {config['out_dir']}")
    ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from iteration {iter_num}, best val loss {best_val_loss}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params/1e6:.2f}M")


def scale_hyperparams(current_step: int, max_steps: int, hyper_param: float, scores: torch.Tensor = None, min_value: float = None) -> float:
    if current_step >= max_steps:
        return hyper_param
    
    if scores is None:
        scaled_value = hyper_param * ((current_step + 1)/ max_steps) ** 1
    else:
        top_p = 0.10
        scaled_value = torch.quantile(scores, 1.0 - top_p).item()

    if min_value is not None:
        scaled_value = max(scaled_value, min_value)
    return scaled_value
    
running_mfu = -1.0
t0 = time.time()

local_iter_num = 0
raw_model = model.module if ddp else model

with tqdm(total=config['max_iters'], desc="Training Progress") as pbar:
    train_losses = []
    iter_num = 0
    scores = None
    for X, Y in train_loader:
        all_z_H, all_z_L = None, None
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        B = X.shape[0]
        active_mask = torch.ones(B, dtype=torch.bool, device=device)
        correctness_threshold = scale_hyperparams(iter_num, config['max_iters'], config.get('correctness_threshold', 0.8), scores=scores, min_value = 0.3)
        if (iter_num % 1000) == 0:
            print(correctness_threshold)
        halt_threshold = scale_hyperparams(iter_num, config['max_iters'], config.get('halt_threshold', 0.9))#0.7))
        # correctness_threshold = config.get('correctness_threshold', 0.8)
        # halt_threshold = config.get('halt_threshold', 0.9)#0.7))
        for ind in range(N_supervised_steps):
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

            lr = get_lr(iter_num, config) if config['decay_lr'] else config['learning_rate']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if ddp:
                model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
            with ctx:
                logits, loss, z_H_new, z_L_new, q_halt_logits = model(X_active, Y_active, z_H_active, z_L_active)
                if all_z_H is None:
                    all_z_H = torch.empty(
                        (B, *z_H_new.shape[1:]), device=z_H_new.device, dtype=z_H_new.dtype
                    )
                    all_z_L = torch.empty(
                        (B, *z_L_new.shape[1:]), device=z_L_new.device, dtype=z_L_new.dtype
                    )            
                all_z_H[active_mask] = z_H_new
                all_z_L[active_mask] = z_L_new
                
                q_loss, halt_now_active, scores = compute_trm_losses_and_halt(
                        logits, Y_active, q_halt_logits, 
                        correctness_threshold=correctness_threshold,
                        halt_threshold=halt_threshold,
                        lambda_q=config.get('lambda_q', 0.1),
                    )
                # loss = (loss + q_loss)/ config['gradient_accumulation_steps']
                # active_mask[active_idx] = active_mask[active_idx] & (~halt_now_active)
            scaler.scale(loss).backward()


            if config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item() * config['gradient_accumulation_steps'])
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if (iter_num + 1) % config['eval_interval'] == 0 and master_process:
            losses = estimate_loss(model, val_loader, device, config, ctx)
            print(f"\nStep {iter_num}: train loss {np.mean(train_losses):.4f}, val loss {losses['val']:.4f} | val_bpc {losses['val'] / math.log(2):8.3f}")
            train_losses = []
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': get_serializable_config(config),
                }
                checkpoint_path = os.path.join(config['out_dir'], 'ckpt.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        iter_num += 1
        local_iter_num += 1
        pbar.update(1)

        if iter_num % config['log_interval'] == 0 and master_process:
            lossf = loss.item() * config['gradient_accumulation_steps']
            print(f"Iter {iter_num}: loss {lossf:5.2f} | ppl {math.exp(lossf):8.2f} | bpc {lossf / math.log(2):8.3f}")
        if iter_num > config['max_iters']:
            break                