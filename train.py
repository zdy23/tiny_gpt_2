"""针对 A100 GPU 优化的小型 GPT-2 训练脚本"""

import math
import time
import csv
import itertools
import concurrent.futures
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GPT
from dataset import TextDataset
from tokenizer import Tokenizer


class Config:
    vocab_size: int = 50257
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    batch_size: int = 64
    micro_batch_size: int = 16
    grad_accum_steps: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 30000
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000
    val_steps: int = 50
    data_file: str = "data/TinyStories-train.txt"
    num_workers: int = 8
    shuffle: bool = True
    dtype: str = "bfloat16"
    compile: bool = True
    checkpoint: bool = False
    output_dir: str = "checkpoints/"

    def __init__(self):
        if self.grad_accum_steps == 0:
            assert self.batch_size % self.micro_batch_size == 0
            self.grad_accum_steps = self.batch_size // self.micro_batch_size


def log_csv(path, step, train_loss, val_loss, lr, tok_s):
    row = {"step": step, "train_loss": f"{train_loss:.6f}",
           "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
           "lr": f"{lr:.2e}", "tok/s": f"{tok_s:.0f}"}
    p = Path(path)
    first = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if first:
            w.writeheader()
        w.writerow(row)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _encode_chunk(text_chunk):
    import tiktoken
    return tiktoken.get_encoding("gpt2").encode(text_chunk, allowed_special="all")


def create_dataloaders(cfg, tokenizer):
    cache_file = Path("data/tokens_cache.pt")
    data_file = Path(cfg.data_file)

    if cache_file.exists() and cache_file.stat().st_mtime > data_file.stat().st_mtime:
        print("Loading cached tokens ...")
        tokens = torch.load(cache_file, map_location="cpu").tolist()
    else:
        print(f"Loading data ({data_file.stat().st_size / 1e6:.1f} MB) ...")
        with open(data_file, encoding="utf-8") as f:
            lines = f.readlines()
        num_chunks = min(32, len(lines) or 1)
        chunks = ["".join(lines[i::num_chunks]) for i in range(num_chunks)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(cfg.num_workers, 16)) as ex:
            results = list(tqdm(ex.map(_encode_chunk, chunks), total=num_chunks, desc="Tokenizing"))
        tokens = list(itertools.chain.from_iterable(results))
        print(f"Total tokens: {len(tokens):,}")
        torch.save(torch.tensor(tokens), cache_file)

    split = int(len(tokens) * 0.9)
    train_ds = TextDataset(tokens[:split], cfg.max_seq_len)
    val_ds = TextDataset(tokens[split:], cfg.max_seq_len)
    loader_kw = dict(batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=cfg.shuffle, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, cfg, device, dtype):
    model.eval()
    total_loss = 0.0
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.val_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / min(cfg.val_steps, len(val_loader))


def save_checkpoint(model, optimizer, scheduler, step, cfg, loss, is_best=False):
    path = Path(cfg.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt = {"step": step, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "loss": loss,
            "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")}}
    torch.save(ckpt, path / ("best.pt" if is_best else f"step_{step}.pt"))
    torch.save(ckpt, path / "latest.pt")


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    print(f"Device: {device}")
    print(f"Output dir: {cfg.output_dir}")

    tokenizer = Tokenizer()
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)

    model = GPT(tokenizer.vocab_size, cfg.d_model, cfg.n_layer, cfg.n_head, cfg.d_ff,
                cfg.max_seq_len, cfg.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if cfg.checkpoint:
        model.gradient_checkpointing_enable()
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Using torch.compile")

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        (no_decay_params if p.ndim < 2 or any(k in name for k in ("bias", "norm", "ln")) else decay_params).append(p)
    optimizer = AdamW([{"params": decay_params, "weight_decay": cfg.weight_decay},
                       {"params": no_decay_params, "weight_decay": 0.0}],
                      lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=True)

    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.total_steps)
    scaler = torch.amp.GradScaler(device) if cfg.dtype == "float16" else None
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    step, total_t0, best_val_loss = 0, time.time(), float("inf")
    train_iter = iter(train_loader)
    progress_bar = tqdm(total=cfg.total_steps, desc="Training")
    tokens_processed = 0

    while step < cfg.total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss = model(x, y)[1] / cfg.grad_accum_steps
            (scaler.scale(loss) if scaler else loss).backward()
            accum_loss += loss.item()
            tokens_processed += x.numel()

        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        (scaler.step(optimizer) and scaler.update()) if scaler else optimizer.step()
        scheduler.step()
        step += 1
        progress_bar.update(1)

        if step % cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            tok_s = tokens_processed / (time.time() - total_t0)
            print(f"step {step:>6d} | loss {accum_loss:.4f} | lr {lr:.2e} | tok/s {tok_s:.0f}")
            log_csv(f"{cfg.output_dir}metrics.csv", step, accum_loss, None, lr, tok_s)

        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, cfg, device, dtype)
            lr = scheduler.get_last_lr()[0]
            tok_s = tokens_processed / (time.time() - total_t0)
            print(f"step {step:>6d} | val loss {val_loss:.4f}")
            tqdm.write(f"step {step:>6d} | val loss {val_loss:.4f}")
            log_csv(f"{cfg.output_dir}metrics.csv", step, accum_loss, val_loss, lr, tok_s)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, step, cfg, val_loss, is_best=True)

        if step % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, step, cfg, accum_loss)

    progress_bar.close()
    save_checkpoint(model, optimizer, scheduler, step, cfg, accum_loss)
    t_total = time.time() - total_t0
    print(f"\nDone! {t_total:.1f}s, Tokens/s: {tokens_processed / t_total:.0f}")


if __name__ == "__main__":
    main()
