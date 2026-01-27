import math

# Configuration for the modified model
out_dir = 'out-enwik8-char-rope-trmdeep4nsup5'  # Output directory for model checkpoints and logs

always_save_checkpoint = True  # Ensure we save checkpoints
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-rope'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 128 # Adjust based on your GPU memory
block_size = 256 # Context length

# Model parameters
n_layer = 8
n_head = 8
n_embd = 512#256#512
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
N_supervised_steps = 6
N_supervised_steps_eval = N_supervised_steps - 2  # Use one less step during evaluation

# Use the modified model
model_type = 'trm'

# System parameters
device = 'cuda'  # Use CUDA for training
dtype = 'float16'  # Use float16 for faster training
compile = False  # Disable compilation for now