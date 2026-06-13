import torch.nn as nn
from .embedding import ModelEmbedding
from .norm import LayerNorm
from .block import Block


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.embedding = ModelEmbedding(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.token_emb.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, nn.functional.cross_entropy(logits.view(B * T, C), targets.view(B * T))
