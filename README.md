# Tiny GPT-2

A scratch-built GPT-2-style autoregressive language model implemented with PyTorch, trained on the TinyStories dataset.

## Project Structure

```
tiny_gpt_2/
├── model/               # Transformer Decoder building blocks
│   ├── embedding.py     # Token + learned Position Embedding
│   ├── attention.py     # Causal Multi-Head Self-Attention (Flash Attention)
│   ├── mlp.py           # Two-layer MLP with GELU activation
│   ├── norm.py          # LayerNorm
│   ├── block.py         # Pre-Norm Transformer Block
│   └── gpt.py           # GPT model assembly + weight tying
├── tokenizer.py         # GPT-2 tokenizer (tiktoken)
├── dataset.py           # Sliding-window next-token prediction Dataset
├── train.py             # Training script (A100-optimized)
├── eval.py              # Evaluation script (perplexity)
├── test.py              # Interactive text generation
├── plot_loss.py         # Loss curve visualization
└── data/                # TinyStories training data
```

## Model Configuration

| Config | Value |
|---|---:|
| d_model | 768 |
| n_layer | 12 |
| n_head | 12 |
| d_ff | 3072 |
| max_seq_len | 1024 |
| Parameters | 124.4M |

## Training

```bash
python train.py
```

Features:
- AdamW (fused) + Linear warmup + Cosine decay
- Gradient accumulation (global batch size = 64)
- bfloat16 mixed precision, `torch.compile`, Flash Attention
- Automatic token caching for faster re-runs
- CSV logging + loss curve plotting
- Best / latest checkpoint auto-save

## Evaluation

```bash
python eval.py                        # loads best.pt by default
python eval.py --checkpoint checkpoints/step_10000.pt
```

## Generation

```bash
python test.py
```

Interactive story generation with temperature, top-k, top-p sampling, and repetition penalty.

## Visualization

```bash
python plot_loss.py
```

## Dependencies

- Python >= 3.11
- PyTorch >= 2.12
- tiktoken, numpy, tqdm, matplotlib, pandas

## Results

| Metric | Value |
|---|---:|
| Final Train Loss | ≈ 1.77 |
| Validation Loss | ≈ 1.60 |
| Perplexity | ≈ 4.95 |
| Throughput | ≈ 165k tokens/s (A100) |
