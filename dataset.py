import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    # 用滑动窗口切分数据：输入 x 预测下一个 token y

    def __init__(self, tokens: list[int], max_seq_len: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        # 总 token 数减去窗口长度就是样本数
        return len(self.tokens) - self.max_seq_len

    def __getitem__(self, idx: int):
        # x 取 [idx, idx+seq_len)，y 取 [idx+1, idx+seq_len+1)（右移一位）
        return (self.tokens[idx : idx + self.max_seq_len],
                self.tokens[idx + 1 : idx + self.max_seq_len + 1])
