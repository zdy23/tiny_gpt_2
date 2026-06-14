import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    # 多头因果自注意力：每个 token 只能看到自己及之前的 token

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Q、K、V 的线性变换（无偏置）
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x):
        # (B, T, C) -> 拆成多头 -> (B, n_head, T, head_dim)
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # PyTorch 内置的高效因果注意力计算
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        # 合并多头结果并映射回原始维度
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))
