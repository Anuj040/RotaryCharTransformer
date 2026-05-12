#!/usr/bin/env python3
"""
Prepare a byte-level mixed-domain training set:
  train_byte.bin = enwik8 train  |  tinyshakespeare  |  openwebtext (streamed)
  val_byte.bin   = enwik8 val    (copied verbatim)

Target total train ~= 4 x current enwik8 train (~90 MB) => ~360 MB bytes.
Each byte is one token (vocab_size = 256) stored as uint16 to match the
existing dataloader (src/utils/data_utils/prepare_dataset.py).
"""
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset
from tqdm import tqdm

# ---- paths ----------------------------------------------------------------
HERE = Path(__file__).parent
ENWIK8_DIR = HERE.parent / "enwik8"

TRAIN_OUT = HERE / "train_byte.bin"
VAL_OUT = HERE / "val_byte.bin"
META_OUT = HERE / "meta_byte.pkl"
SHAKE_TXT = HERE / "tinyshakespeare.txt"

# ---- budget ---------------------------------------------------------------
ENWIK8_TRAIN_BYTES = (ENWIK8_DIR / "train_byte.bin").stat().st_size // 2  # uint16 -> bytes
TARGET_TOTAL_BYTES = 4 * ENWIK8_TRAIN_BYTES                               # ~360 MB

SHAKE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# ---- 1) enwik8 train (already byte-level uint16) --------------------------
print(f"[1/4] enwik8 train: {ENWIK8_TRAIN_BYTES:,} bytes")
enwik8_train = np.fromfile(ENWIK8_DIR / "train_byte.bin", dtype=np.uint16)

# ---- 2) tinyshakespeare ---------------------------------------------------
if not SHAKE_TXT.exists():
    print(f"[2/4] downloading tinyshakespeare -> {SHAKE_TXT}")
    SHAKE_TXT.write_bytes(requests.get(SHAKE_URL).content)
shake_bytes = SHAKE_TXT.read_bytes()
print(f"[2/4] tinyshakespeare: {len(shake_bytes):,} bytes")
shake_arr = np.frombuffer(shake_bytes, dtype=np.uint8).astype(np.uint16)

# ---- 3) openwebtext stream until budget hit -------------------------------
owt_budget = TARGET_TOTAL_BYTES - ENWIK8_TRAIN_BYTES - len(shake_bytes)
assert owt_budget > 0, "budget already exhausted before OWT"
print(f"[3/4] streaming openwebtext, target {owt_budget:,} bytes")

owt_chunks: list[np.ndarray] = []
owt_written = 0
SEP = b"\n\n"  # separator between OWT documents

ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
pbar = tqdm(total=owt_budget, unit="B", unit_scale=True, desc="owt bytes")
for ex in ds:
    b = ex["text"].encode("utf-8", errors="replace") + SEP
    if owt_written + len(b) > owt_budget:
        b = b[: owt_budget - owt_written]  # truncate the last doc to land exactly on budget
    owt_chunks.append(np.frombuffer(b, dtype=np.uint8).astype(np.uint16))
    owt_written += len(b)
    pbar.update(len(b))
    if owt_written >= owt_budget:
        break
pbar.close()
owt_arr = np.concatenate(owt_chunks) if owt_chunks else np.empty(0, dtype=np.uint16)
print(f"[3/4] openwebtext: {len(owt_arr):,} bytes from {len(owt_chunks):,} documents")

# ---- 4) concat + write ----------------------------------------------------
train_arr = np.concatenate([enwik8_train, shake_arr, owt_arr])
print(
    f"[4/4] writing train_byte.bin "
    f"(enwik8={len(enwik8_train):,} + shake={len(shake_arr):,} + owt={len(owt_arr):,} "
    f"= {len(train_arr):,} bytes)"
)
train_arr.tofile(TRAIN_OUT)

# val: copy enwik8 val_byte.bin verbatim
shutil.copyfile(ENWIK8_DIR / "val_byte.bin", VAL_OUT)
val_len = VAL_OUT.stat().st_size // 2
print(f"[4/4] val_byte.bin copied from enwik8 ({val_len:,} bytes)")

with open(META_OUT, "wb") as f:
    pickle.dump({"vocab_size": 256}, f)
print(f"[4/4] wrote meta_byte.pkl (vocab_size=256)")

print("done.")
print(f"  {TRAIN_OUT}  -> {TRAIN_OUT.stat().st_size:,} B on disk (uint16)")
print(f"  {VAL_OUT}    -> {VAL_OUT.stat().st_size:,} B on disk (uint16)")