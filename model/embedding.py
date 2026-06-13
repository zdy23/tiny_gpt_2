import torch.nn as nn
import torch


class ModelEmbedding(nn.Module):
    """Token 嵌入 + 可学习位置编码"""

    def __init__(self, vocab_size, d_model, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        return self.token_emb(x) + self.pos_emb(torch.arange(x.size(1), device=x.device).unsqueeze(0))
