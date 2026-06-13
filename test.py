"""交互式故事生成"""

import torch
from model import GPT
from tokenizer import Tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_seq_len=1024, max_new_tokens=200,
             temperature=0.4, top_k=3, top_p=0.85, rep_penalty=1.05, device="cuda"):
    # 编码提示词，初始化为 (1, prompt_len)
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    eos_id = tokenizer.eos_id

    for _ in range(max_new_tokens):
        # 截取不超过 max_seq_len 的上下文窗口
        x = input_ids[:, -min(max_seq_len, input_ids.shape[1]):]
        # 取最后一个位置的 logits 并施加温度
        logits = model(x)[0][:, -1, :] / temperature

        # 重复惩罚: 已出现的 token 降低概率
        if rep_penalty != 1.0:
            for tid in input_ids[0].tolist():
                logits[0, tid] /= rep_penalty if logits[0, tid] > 0 else (1 / rep_penalty)

        # top-k: 只保留概率最高的 k 个 token
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][:, -1:]] = float("-inf")

        # top-p 核采样: 累积概率超过 p 的 token 被过滤
        if top_p < 1.0:
            sorted_lg, sorted_idx = torch.sort(logits, descending=True)
            cumsum = torch.cumsum(torch.softmax(sorted_lg, dim=-1), dim=-1)
            sorted_lg[cumsum - sorted_lg.softmax(-1) >= top_p] = float("-inf")
            logits = sorted_lg.gather(1, sorted_idx.argsort(-1))

        # 从概率分布中采样下一个 token
        next_id = torch.multinomial(logits.softmax(-1), 1)
        if next_id.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载训练好的检查点
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
    cfg = ckpt["config"]

    model = GPT(cfg["vocab_size"], cfg["d_model"], cfg["n_layer"], cfg["n_head"],
                cfg["d_ff"], cfg["max_seq_len"], dropout=0.0)
    # 去掉 torch.compile 添加的前缀
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = Tokenizer()
    print("Interactive Generation (temp=0.4 top_k=3 top_p=0.85 rep=1.05)")
    print("Type a prompt. Ctrl+C or 'quit' to exit.\n")

    # 交互式循环: 读入提示 -> 生成故事 -> 打印
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
