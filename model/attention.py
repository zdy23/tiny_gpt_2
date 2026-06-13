import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头因果自注意力"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))
