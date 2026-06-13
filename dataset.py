import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """滑动窗口下一个 token 预测的数据集"""

    def __init__(self, tokens: list[int], max_seq_len: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.max_seq_len

    def __getitem__(self, idx: int):
        return (self.tokens[idx : idx + self.max_seq_len],
                self.tokens[idx + 1 : idx + self.max_seq_len + 1])
