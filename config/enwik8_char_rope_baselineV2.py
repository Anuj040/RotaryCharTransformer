import math
# Configuration for RoPE model with baseline hyperparameters
out_dir = 'out-enwik8-char-rope-baselineV2'  # New output directory

always_save_checkpoint = True
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-rope-baseline'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

# Model parameters (copied from baseline)
n_layer = 12
n_head = 8
n_embd = 384
freq = 10000  # Frequency parameter for RoPE
dropout = 0.1
bias = False

# Optimization parameters (copied from baseline)
max_iters = 4000
learning_rate = 1e-3 * (5000 / max_iters)
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
init_from = 'scratch'
N_supervised_steps = 6
N_supervised_steps_eval = N_supervised_steps - 1

# Use the RoPE model
model_type = 'nope'

# System parameters
device = 'cuda'
dtype = 'float16'
compile = False
