"""Framework-free word tokenizer for the CNN.

One `encode` function is used for training AND inference, always with post-padding and
post-truncation. (The original notebook trained with padding='post' but predicted with
Keras' default padding='pre', which silently changed the model input.)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PAD_ID, OOV_ID = 0, 1
_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


class WordTokenizer:
    def __init__(self, max_words: int = 30000):
        self.max_words = max_words
        self.index: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.index) + 2

    @staticmethod
    def tokens(text: str) -> list[str]:
        return _TOKEN.findall(str(text).lower())

    def fit(self, texts) -> "WordTokenizer":
        counts: Counter = Counter()
        for t in texts:
            counts.update(self.tokens(t))
        most_common = counts.most_common(self.max_words - 2)
        self.index = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        return self

    def encode(self, texts, maxlen: int) -> np.ndarray:
        texts = list(texts)
        arr = np.full((len(texts), maxlen), PAD_ID, dtype="int32")
        for row, t in enumerate(texts):
            ids = [self.index.get(w, OOV_ID) for w in self.tokens(t)[:maxlen]]
            arr[row, : len(ids)] = ids
        return arr

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"max_words": self.max_words, "index": self.index}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "WordTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(d["max_words"])
        tok.index = d["index"]
        return tok
