# Tiny GPT-2

一个从零手写的 GPT-2 风格自回归语言模型，基于 PyTorch 实现，在 TinyStories 数据集上训练。

我写这个项目是为了让更多人能直观地理解 Transformer 和 GPT 的每一个细节——不用翻论文、不用啃复杂的框架代码，这里每一行都很简单，每个文件都很短，任何学过 Python 的人都能看懂。

## 效果预览

```bash
>>> Once upon a time, there was a little dragon
Once upon a time, there was a little dragon named Sparky. He lived in a big cave with his mom and dad.
Sparky loved to play with his friends, but one day he felt very sad...
```

这是一个只有 124M 参数的"小"模型，在一台 A100 上训练几个小时后，已经能写出像样的故事了。当然，它和 ChatGPT 完全没法比，但麻雀虽小，五脏俱全。

## 项目结构

```
tiny_gpt_2/
├── model/
│   ├── embedding.py     # 词嵌入 + 可学习的位置编码
│   ├── attention.py     # 多头因果自注意力（Flash Attention）
│   ├── mlp.py           # 两层 MLP + GELU 激活
│   ├── norm.py          # LayerNorm
│   ├── block.py         # Pre-Norm 的 Transformer 解码器块
│   └── gpt.py           # 组装 GPT 模型 + 权重共享
├── tokenizer.py         # GPT-2 分词器（基于 tiktoken）
├── dataset.py           # 滑动窗口的方式准备训练数据
├── train.py             # 训练脚本（针对 A100 优化）
├── eval.py              # 评估脚本（计算困惑度）
├── test.py              # 交互式故事生成
├── plot_loss.py         # 画损失曲线
└── data/                # TinyStories 训练数据（需自行下载）
```

整个模型的核心代码不超过 200 行。如果你正在学 Transformer，这个项目是很好的起点——读一遍，比看三天博客都有用。

## 模型配置

| 配置项 | 值 |
|---|---|
| d_model（隐藏维度） | 768 |
| n_layer（层数） | 12 |
| n_head（注意力头数） | 12 |
| d_ff（前馈网络维度） | 3072 |
| max_seq_len（最大序列长度） | 1024 |
| 参数量 | 124.4M |

这基本就是一个小号 GPT-2。如果你有更好的显卡，把 `d_model` 改成 768 往大了调就行。

## 快速开始

### 1. 安装依赖

```bash
pip install torch tiktoken numpy tqdm matplotlib pandas
```

需要 Python >= 3.11，PyTorch >= 2.0（推荐 2.x 以使用 `torch.compile` 和 Flash Attention）。

### 2. 准备数据

把 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的训练文本放到 `data/TinyStories-train.txt`。

### 3. 训练

```bash
python train.py
```

脚本会自动分词并缓存，之后重复运行会直接读缓存，快很多。默认会训练 30000 步，约在一张 A100 上跑 2-3 小时。

训练中做的事情：
- AdamW 优化器 + 余弦退火学习率（前 2000 步预热）
- 梯度累积实现等效 batch size = 64
- bfloat16 混合精度训练 + `torch.compile` + Flash Attention
- 每 500 步在验证集上评估，自动保存最优模型
- 训练日志写入 `checkpoints/metrics.csv`

### 4. 评估

```bash
python eval.py                         # 默认加载 best.pt
python eval.py --checkpoint checkpoints/step_10000.pt
python eval.py --quick                 # 快速评估（只读前 20000 行）
```

输出困惑度（perplexity），数字越低越好。

### 5. 生成

```bash
python test.py
```

交互式输入提示词，模型续写故事。生成用到了：
- **温度（temperature=0.4）**：控制随机性，越低越"保守"
- **top-k（k=3）**：只从概率最高的 3 个 token 中采样
- **top-p（p=0.85）**：核采样，累积概率到 85% 就截断
- **重复惩罚（1.05）**：轻微降低已出现 token 的概率

前 50 步强制不输出结束符，之后也会抑制它，所以模型通常会讲一个完整的小故事再停下来。

### 6. 可视化

```bash
python plot_loss.py
```

画一张带平滑曲线、验证损失、标准差带的精美图表，保存到 `checkpoints/loss_curve.png`。

## 训练结果

| 指标 | 数值 |
|---|---|
| 最终训练损失 | ≈ 1.77 |
| 验证损失 | ≈ 1.60 |
| 困惑度 | ≈ 4.95 |
| 训练速度 | ≈ 165k tokens/s（A100） |

## 代码导读

如果你想知道从哪开始看，建议按这个顺序：

1. **`model/gpt.py`**——整个模型的入口，看 `forward` 怎么把各个组件串起来
2. **`model/block.py`**——一个 Transformer 块：先做注意力，再做前馈，中间夹着残差连接
3. **`model/attention.py`**——最核心的部分：Q、K、V 怎么来的，因果掩码怎么确保"只看过去"
4. **`tokenizer.py`**——文本怎么变成数字，数字怎么变回文本
5. **`train.py`**——训练循环：梯度累积、混合精度、学习率调度都在这里
6. **`test.py`**——生成过程：温度、top-k、top-p 到底怎么影响输出

整个项目是一个"玩具级"的 GPT，但 tokenizer 用的是真 GPT-2 的分词器（50257 个 token），模型结构也和真 GPT-2 一样。看懂这个，看懂 GPT-2 的论文就没什么障碍了。

## 感谢

- [Andrej Karpathy](https://github.com/karpathy) 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [minbpe](https://github.com/karpathy/minbpe) 是重要的灵感来源
- [TinyStories](https://arxiv.org/abs/2305.07759) 论文提供了绝佳的小规模训练数据
- OpenAI 开源的 [tiktoken](https://github.com/openai/tiktoken) 让分词变得简单

## License

MIT
