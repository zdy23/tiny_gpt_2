import tiktoken

SPECIAL_TOKEN = "<|endoftext|>"


class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self._eos_id = self.enc.encode_single_token(SPECIAL_TOKEN)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._eos_id

    @property
    def bos_id(self) -> int:
        return self._eos_id
