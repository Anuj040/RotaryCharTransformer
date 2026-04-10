import torch


class EnwikDataset(torch.utils.data.Dataset):
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
