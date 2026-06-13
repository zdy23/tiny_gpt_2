import torch.nn as nn


class MLP(nn.Module):
    """逐位置前馈网络"""

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))
