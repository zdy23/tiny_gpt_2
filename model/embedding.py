import torch.nn as nn
import torch


class ModelEmbedding(nn.Module):
    # 把 token 编号转成向量（词嵌入），再加上位置编码

    def __init__(self, vocab_size, d_model, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        # 词嵌入 + 位置嵌入，位置从 0 到序列长度-1
        return self.token_emb(x) + self.pos_emb(torch.arange(x.size(1), device=x.device).unsqueeze(0))
