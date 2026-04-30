import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from src.utils.model_utilities.pick_model import select_model

from src.utils.eval_utils.loss_fn import estimate_test_loss

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
    model = select_model(config)
    model.to(device)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['out_dir'], "ckpt.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    test_data_path = os.path.join(data_dir, 'test.bin')
    test_data = np.memmap(test_data_path, dtype=np.uint16, mode='r')
    test_data = torch.from_numpy(test_data.astype(np.int64))

    block_size = config['block_size']
    batch_size = config.get('batch_size', 64)
    N_supervised_steps = config.get('N_supervised_steps_eval', 2)
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

    estimate_test_loss(model,  val_loader, N_supervised_steps, device, ctx)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--checkpoint', type=str, required=False, help='Checkpoint file')
    args = parser.parse_args()
    main(args.config, args.checkpoint)
