import torch.nn as nn
from .norm import LayerNorm
from .attention import MultiHeadAttention
from .mlp import MLP


class Block(nn.Module):
    # 一个 Transformer 解码器块：先做注意力，再做前馈，每步都有残差连接

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        # 注意力子层 + 残差
        x = x + self.attn(self.ln1(x))
        # 前馈子层 + 残差
        x = x + self.mlp(self.ln2(x))
        return x
