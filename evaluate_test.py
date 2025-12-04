import os
import math
import pickle
import numpy as np
import torch
from contextlib import nullcontext

from model import GPTConfig
from model_baseline import BaselineGPT
from model_rope import GPTWithRoPE

def get_batch(data, config, device, device_type):
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def main(config_file, checkpoint_path):
    config = {}
    with open(config_file, 'r') as f:
        exec(f.read(), {}, config)
    config = {k: v for k, v in config.items() if not k.startswith('__')}
    device = config['device']
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    data_dir = os.path.join('data', config['dataset'])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    config['vocab_size'] = vocab_size

    gpt_config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']
    gpt_config = {k: v for k, v in config.items() if k in gpt_config_keys}
    gptconf = GPTConfig(**gpt_config)

    if config.get('model_type') in ('rope', 'nope'):
        model = GPTWithRoPE(gptconf)
        print("Using GPTWithRoPE model.")
    elif config.get('model_type') == 'trm':
        from model_rope_trm import TRMGPTWithRoPE
        model = TRMGPTWithRoPE(gptconf)
        print("Using TRMGPTWithRoPE model.")
    else:
        model = BaselineGPT(gptconf)
        print("Using BaselineGPT model.")
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    test_data_path = os.path.join(data_dir, 'test.bin')
    test_data = np.memmap(test_data_path, dtype=np.uint16, mode='r')
    test_data = torch.from_numpy(test_data.astype(np.int64))

    block_size = config['block_size']
    batch_size = config.get('batch_size', 64)
    num_tokens = len(test_data) - 1
    x_tokens = test_data[:num_tokens]
    y_tokens = test_data[1:num_tokens+1]
    num_batches = num_tokens // block_size
    x_tokens = x_tokens[:num_batches * block_size]
    y_tokens = y_tokens[:num_batches * block_size]
    x_batches = x_tokens.view(-1, block_size)
    y_batches = y_tokens.view(-1, block_size)

    val_dataset = torch.utils.data.TensorDataset(x_batches, y_batches)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def estimate_test_loss():
        losses = []
        for X, Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        bpc = mean_loss / math.log(2)
        print(f"Test loss: {mean_loss:.4f} | test_bpc: {bpc:8.3f}")
        return mean_loss, bpc

    estimate_test_loss()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file')
    args = parser.parse_args()
    main(args.config, args.checkpoint)
