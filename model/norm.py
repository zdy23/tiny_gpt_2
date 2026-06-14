import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    # 层归一化：对每个样本的最后一维做标准化，再缩放平移

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # gamma 和 beta 是可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma + self.beta
