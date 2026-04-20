# Configuration for RoPE model with baseline hyperparameters
out_dir = 'out-enwik8-char-rope-baseline'  # New output directory

eval_interval = 500
eval_iters = 200
log_interval = 100

always_save_checkpoint = True
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-rope-baseline'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Model parameters (copied from baseline)
n_layer = 12
n_head = 8
n_embd = 384
dropout = 0.1
bias = False

# Optimization parameters (copied from baseline)
learning_rate = 1e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
init_from = 'scratch'

# Use the RoPE model
model_type = 'rope'

# System parameters
device = 'cuda'
dtype = 'float16'
compile = False
