import torch


# Configuration for the modified model
out_dir = 'outputs/sup3_2deep_2lyrxsainitperlyr_accum'  # Output directory for model checkpoints and logs

always_save_checkpoint = True  # Ensure we save checkpoints
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = out_dir.split("/")[-1]

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 128 # Adjust based on your GPU memory
block_size = 256 # Context length

# Model parameters
n_layer = 8
n_head = 8
n_embd = 512
freq = 10000  # Frequency parameter for RoPE
dropout = 0.1  # Added some dropout for regularization
bias = False  # No bias in LayerNorm and Linear layers
perlayerembeds = False
num_recursive_steps = 4
num_deep_recursions = 2

# Optimization parameters
max_iters = 8000#2500  # Number of iterations for training
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
N_supervised_steps = 3
N_supervised_steps_eval = max(N_supervised_steps - 5, 3)  # Use one less step during evaluation

# Use the modified model
model_type = 'trm'

# System parameters
device = 'cuda'  if torch.cuda.is_available() else "cpu" # Use CUDA for training
dtype = 'float16'  if torch.cuda.is_available() else "float32" # Use float16 for faster training