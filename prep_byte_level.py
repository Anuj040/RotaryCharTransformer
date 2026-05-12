#!/usr/bin/env python3
"""
Prepare byte-level enwik8 bin files.
Each raw byte becomes one token (vocab_size=256).
Output format matches the existing char-level files (uint16 arrays).
"""
import numpy as np
import pickle
from pathlib import Path

data_dir = Path(__file__).parent

splits = [
    ("train", "train.txt.raw", "train_byte.bin"),
    ("val",   "valid.txt.raw", "val_byte.bin"),
    ("test",  "test.txt.raw",  "test_byte.bin"),
]

for split, raw_file, out_file in splits:
    raw_path = data_dir / raw_file
    out_path = data_dir / out_file
    with open(raw_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    data.astype(np.uint16).tofile(out_path)
    print(f"{split:5s}: {len(data):>12,} byte-tokens  →  {out_file}")

meta = {"vocab_size": 256}
with open(data_dir / "meta_byte.pkl", "wb") as f:
    pickle.dump(meta, f)
print("Saved meta_byte.pkl  (vocab_size=256)")
