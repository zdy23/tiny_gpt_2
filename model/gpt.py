import torch.nn as nn
from .embedding import ModelEmbedding
from .norm import LayerNorm
from .block import Block


class GPT(nn.Module):
    # 完整的 GPT 模型：嵌入 -> N 个解码器块 -> 输出层

    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.embedding = ModelEmbedding(vocab_size, d_model, max_seq_len)
        # 堆叠 N 层 Transformer 块
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(d_model)
        # 输出层，将隐藏状态映射回词表大小
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 共享词嵌入和输出层的权重（weight tying）
        self.embedding.token_emb.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        # 最后做一次层归一化，再映射到词表得到 logits
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        # 计算交叉熵损失：把 (B, T, C) 展平成 (B*T, C) 再算
        B, T, C = logits.shape
        return logits, nn.functional.cross_entropy(logits.view(B * T, C), targets.view(B * T))
