import os
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EnwikDataset(Dataset):
    def __init__(self, data_path: str, block_size: int) -> None:
        data = np.fromfile(data_path, dtype=np.uint16)
        self.data = torch.from_numpy(data.astype(np.int64))
        self.block_size = block_size
        # self.block_size = block_size // 2

        # number of full non-overlapping blocks
        self.n_blocks = len(self.data) // self.block_size

    def __len__(self) -> int:
        return self.n_blocks - 1  # since y starts one block after x
        # return self.n_blocks - 2  # since y starts one block after x

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Each idx corresponds to a full block
        start = idx * self.block_size
        end = start + self.block_size
        # end = start + self.block_size + self.block_size

        x = self.data[start:end]
        y = self.data[
            start + 1 : end + 1
        ]  # still next-token prediction inside the block

        return x, y


class EnwikDatasetV2(Dataset):
    def __init__(self, data_path: str, block_size: int) -> None:
        data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.data = torch.from_numpy(data.astype(np.int64))
        self.block_size = block_size
        # number of full non-overlapping blocks
        self.n_blocks = len(self.data) // self.block_size

    def __len__(self) -> int:
        return self.n_blocks - 1  # since y starts one block after x
        # return self.n_blocks - 2  # since y starts one block after x

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.data) - self.block_size, ())
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def get_dataloaders(config, device_type: str) -> tuple[DataLoader]:

    data_dir = os.path.join("data", config["dataset"])
    _sfx = "_byte" if config.get("encoding", "char") == "byte" else ""

    train_dataset = EnwikDataset(
        os.path.join(data_dir, f"train{_sfx}.bin"), config["block_size"]
    )
    val_dataset = EnwikDataset(
        os.path.join(data_dir, f"val{_sfx}.bin"), config["block_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        prefetch_factor=4,
        pin_memory=(device_type == "cuda"),
        num_workers=4 if device_type == "cuda" else 2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2 * config["batch_size"],
        shuffle=False,
        prefetch_factor=8,
        pin_memory=(device_type == "cuda"),
        num_workers=4 if device_type == "cuda" else 2,
    )
    train_loader = cycle(train_loader)
    return train_loader, val_loader
