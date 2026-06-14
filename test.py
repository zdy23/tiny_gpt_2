"""交互式故事生成"""

import torch
from model import GPT
from tokenizer import Tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_seq_len=1024, max_new_tokens=200,
             temperature=0.4, top_k=3, top_p=0.85, rep_penalty=1.05, device="cuda"):
    # 把提示词编码成 token 编号，作为生成起点
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    eos_id = tokenizer.eos_id

    input_len = input_ids.shape[1]
    for step in range(max_new_tokens):
        # 只取最近 max_seq_len 个 token 作为上下文
        x = input_ids[:, -min(max_seq_len, input_ids.shape[1]):]
        # 取最后一个位置的输出 + 除以温度（温度越低越确定）
        logits = model(x)[0][:, -1, :] / temperature

        # 重复惩罚：已经生成的 token 概率打折
        if rep_penalty != 1.0:
            for tid in input_ids[0].tolist():
                logits[0, tid] /= rep_penalty if logits[0, tid] > 0 else (1 / rep_penalty)

        # top-k 过滤：只保留概率最高的 k 个候选
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][:, -1:]] = float("-inf")

        # top-p 核采样：只保留累积概率不超过 p 的候选
        if top_p < 1.0:
            sorted_lg, sorted_idx = torch.sort(logits, descending=True)
            cumsum = torch.cumsum(torch.softmax(sorted_lg, dim=-1), dim=-1)
            sorted_lg[cumsum - sorted_lg.softmax(-1) >= top_p] = float("-inf")
            logits = sorted_lg.gather(1, sorted_idx.argsort(-1))

        # 前 50 步禁止结束，之后降低结束概率，强制生成长文本
        if step < 50:
            logits[0, eos_id] = float("-inf")
        else:
            logits[0, eos_id] -= 3.0

        # 按 softmax 后的概率随机采样下一个 token
        next_id = torch.multinomial(logits.softmax(-1), 1)
        if next_id.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
    cfg = ckpt["config"]

    model = GPT(cfg["vocab_size"], cfg["d_model"], cfg["n_layer"], cfg["n_head"],
                cfg["d_ff"], cfg["max_seq_len"], dropout=0.0)
    # 去掉 compile 添加的前缀，并兼容旧版 .emb. 命名
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    state_dict = {k.replace(".emb.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = Tokenizer()
    print("Interactive Generation (temp=0.4 top_k=3 top_p=0.85 rep=1.05)")
    print("Type a prompt. Ctrl+C or 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            break
        if not prompt or prompt.lower() == "quit":
            break
        print(f"\n{generate(model, tokenizer, prompt, max_seq_len=cfg['max_seq_len'], device=device)}\n")


if __name__ == "__main__":
    main()
