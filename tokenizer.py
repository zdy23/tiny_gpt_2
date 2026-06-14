import tiktoken

# 文本结束的特殊标记符号
SPECIAL_TOKEN = "<|endoftext|>"


class Tokenizer:
    # 用 GPT-2 的 BPE 分词器做编码和解码

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self._eos_id = self.enc.encode_single_token(SPECIAL_TOKEN)

    def encode(self, text: str) -> list[int]:
        # 把字符串变成 token 编号列表
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        # 把 token 编号列表变回字符串
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        # 词表大小（共 50257 个 token）
        return self.enc.n_vocab

    @property
    def eos_id(self) -> int:
        # 结束标记的编号
        return self._eos_id

    @property
    def pad_id(self) -> int:
        # 用 eos 当 padding 用
        return self._eos_id

    @property
    def bos_id(self) -> int:
        # 也用 eos 当开始标记
        return self._eos_id
