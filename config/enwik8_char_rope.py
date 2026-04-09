import math
import torch

# Configuration for the modified model
out_dir = 'outputs/rop_cmpl'  # Output directory for model checkpoints and logs
eval_interval = 500
eval_iters = 200
log_interval = 100

always_save_checkpoint = True  # Ensure we save checkpoints
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = out_dir.split("/")[-1]

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 128  # Adjust based on your GPU memory
block_size = 256  # Context length

# Model parameters
n_layer = 8
n_head = 8
n_embd = 512
freq = 10000  # Frequency parameter for RoPE
dropout = 0.1  # Added some dropout for regularization
bias = False  # No bias in LayerNorm and Linear layers

# Optimization parameters
max_iters = 4000#2500  # Number of iterations for training
learning_rate = 1e-3 * (5000 / max_iters) # Scaled learning rate
lr_decay_iters = max_iters
eval_interval = max_iters // 5
log_interval = 500
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = int(0.02 * max_iters)  # 2% of max_iters
init_from = 'scratch'  # Initialize model from scratch

# Use the modified model
model_type = 'rope'

# System parameters
device = 'cuda'  if torch.cuda.is_available() else "cpu" # Use CUDA for training
dtype = 'float16'  if torch.cuda.is_available() else "float32" # Use float16 for faster training
compile = False  # Disable compilation for now