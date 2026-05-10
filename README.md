# Character-Level Language Model with Rotary Position Embeddings (TRM)

Character-level language modeling on enwik8 with a TRM-style (Tiny Recursive Model) GPT using Rotary Position Embeddings.

## Project Overview

- **Dataset**: enwik8 (~100M characters from Wikipedia)
- **Task**: Character-level language modeling
- **Primary model**: `TRMGPTWithRoPE` — a recursive variant of a RoPE GPT, defined in [src/utils/model_utilities/model_rope_trm.py](src/utils/model_utilities/model_rope_trm.py)
- **Reference models**: standard GPT in [src/utils/model_utilities/model.py](src/utils/model_utilities/model.py); RoPE GPT in [src/utils/model_utilities/model_rope.py](src/utils/model_utilities/model_rope.py)

## Repository Layout

- [config/](config/) — run configurations (e.g. `enwik8_char_rope_trm.py`)
- [src/pipelines/](src/pipelines/) — training/eval entry points (`train_supervised.py`)
- [src/utils/model_utilities/](src/utils/model_utilities/) — model definitions and `pick_model.select_model`
- [src/utils/data_utils/](src/utils/data_utils/) — `EnwikDataset` and data prep
- [src/utils/eval_utils/](src/utils/eval_utils/) — `estimate_loss` and eval helpers
- [src/utils/train_utils/](src/utils/train_utils/) — `get_lr`, `scale_hyperparams`, etc.
- [evaluate_test_supervised.py](evaluate_test_supervised.py) — fixed test-set evaluation harness
- [data/enwik8/](data/enwik8/) — `train.bin`, `val.bin`, `test.bin`, `meta.pkl`

## Setup

Install requirements (use the project's `requirements.txt`). For initial GCP-based data + env setup:

```
make setup-project-gcp
```

## Data

Data shards (`train.bin`, `val.bin`, `test.bin`) and `meta.pkl` are expected under `data/enwik8/`.
The prep script that produced them is at [data/enwik8/prep_enwik8.py](data/enwik8/prep_enwik8.py).

## Training

Train the TRM-RoPE model with the default config:

```
python3 -m src.pipelines.train_supervised --config config/enwik8_char_rope_trm.py
```

The training loop runs supervised recursive steps controlled by `N_supervised_steps`, evaluates every `max_iters // 5` steps, and breaks after the first eval (single-eval autoresearch protocol).

## Evaluation

Test-set evaluation is performed by the read-only harness:

```
python3 evaluate_test_supervised.py --config config/enwik8_char_rope_trm.py
```

The training run also logs `val_bpc` to stdout/`run.log`:

```
grep "^val_bpc:" run.log
```

## Autoresearch

This branch follows the autoresearch protocol described in [program.md](program.md): isolated experiment branches, fixed eval harness, results tracked in `results.tsv` (untracked).