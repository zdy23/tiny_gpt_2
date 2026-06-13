"""在 CPU 上评估小型 GPT-2 检查点"""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GPT
from tokenizer import Tokenizer


class EvalWindowDataset(Dataset):
    """在指定起始位置取滑动窗口"""

    def __init__(self, tokens, max_seq_len, starts):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.max_seq_len = max_seq_len
        self.starts = torch.as_tensor(starts, dtype=torch.long)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.max_seq_len
        return self.tokens[start:end], self.tokens[start + 1:end + 1]


@torch.no_grad()
def compute_perplexity(model, val_loader, max_batches):
    """计算模型在验证集上的困惑度"""
    model.eval()
    total_loss, n_batches, t0 = 0.0, 0, time.time()
    total = min(max_batches, len(val_loader))
    progress = tqdm(total=total, desc="Evaluating", unit="batch")
    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= total:
            break
        total_loss += model(x, y)[1].item()
        n_batches += 1
        avg_loss = total_loss / n_batches
        progress.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{math.exp(avg_loss):.2f}")
        progress.update(1)
    progress.close()
    if n_batches == 0:
        raise RuntimeError("No eval batches. Increase --max-lines or --eval-tokens.")
    avg_loss = total_loss / n_batches
    return avg_loss, math.exp(avg_loss), time.time() - t0


def tokenize_file(data_file, max_lines=None):
    tokenizer = Tokenizer()
    tokens, t0 = [], time.time()
    with open(data_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            tokens.extend(tokenizer.encode(line))
    return tokens, time.time() - t0


def load_eval_tokens(args):
    # 优先从缓存加载，否则重新分词
    cache_path = Path(args.token_cache)
    use_cache = args.source == "cache" or (args.source == "auto" and cache_path.exists() and not args.quick)
    if use_cache:
        t0 = time.time()
        tokens = torch.load(cache_path, map_location="cpu")
        print(f"Loaded token cache: {cache_path} in {time.time() - t0:.1f}s")
        return tokens, "full token cache"
    max_lines = args.max_lines if args.quick else None
    if max_lines is None:
        print("Tokenizing full text file ...")
    else:
        print(f"Tokenizing up to {max_lines:,} lines ...")
    tokens, t = tokenize_file(args.data_file, max_lines=max_lines)
    print(f"Tokenized {len(tokens):,} tokens in {t:.1f}s")
    return torch.tensor(tokens, dtype=torch.long), "raw text"


def build_eval_starts(num_tokens, max_seq_len, num_windows):
    # 在验证集上均匀采样 num_windows 个起始位置
    max_start = num_tokens - max_seq_len - 1
    if max_start < 0:
        raise RuntimeError(f"Only {num_tokens:,} val tokens, need > max_seq_len={max_seq_len}.")
    num_windows = min(num_windows, max_start + 1)
    if num_windows == 1:
        return torch.tensor([0], dtype=torch.long)
    return torch.linspace(0, max_start, steps=num_windows).long()


def load_model(ckpt_path):
    # 从检查点恢复模型结构和权重
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    loss = ckpt.get("loss", "?")
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    print(f"Checkpoint: {ckpt_path}  step={ckpt.get('step', '?')}  loss={loss:.4f}" if isinstance(loss, (int, float)) else "")
    model = GPT(cfg["vocab_size"], cfg["d_model"], cfg["n_layer"], cfg["n_head"], cfg["d_ff"],
                cfg["max_seq_len"], dropout=0.0)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-file", default="data/TinyStories-train.txt")
    parser.add_argument("--token-cache", default="data/tokens_cache.pt")
    parser.add_argument("--source", choices=["auto", "cache", "text"], default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-tokens", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    # 自动选择检查点路径
    best, latest = Path("checkpoints/best.pt"), Path("checkpoints/latest.pt")
    ckpt_path = args.checkpoint or (str(best) if best.exists() else str(latest) if latest.exists() else str(best))
    model, cfg = load_model(ckpt_path)
    max_seq_len = cfg.get("max_seq_len", 1024)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"  Parameters: {num_params / 1e6:.1f}M\n")
    print("=" * 50)
    print("Perplexity Evaluation")
    print("=" * 50)

    # 加载 token 并取后 10% 作为验证集
    tokens, source_name = load_eval_tokens(args)
    split = int(len(tokens) * 0.9)
    val_tokens = tokens[split:]
    if args.eval_tokens > 0:
        val_tokens = val_tokens[:args.eval_tokens]
    if len(val_tokens) <= max_seq_len:
        raise RuntimeError(f"Only {len(val_tokens):,} val tokens, need > max_seq_len={max_seq_len}.")

    # 均匀采样评估窗口，构造 DataLoader
    num_windows = args.batch_size * args.eval_batches
    starts = build_eval_starts(len(val_tokens), max_seq_len, num_windows)
    print(f"  source={source_name}  total_tokens={len(tokens):,}  eval_windows={len(starts):,}")

    val_ds = EvalWindowDataset(val_tokens, max_seq_len, starts)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    avg_loss, ppl, elapsed = compute_perplexity(model, val_loader, args.eval_batches)
    print(f"  Val Loss: {avg_loss:.4f}  Perplexity: {ppl:.2f}  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
