"""ESM-2's tokenizer, written out, so the runtime needs no transformers.

ESM-2's alphabet is 33 entries and the mapping is one character to one token.
Carrying `transformers` and `tokenizers` to do that costs hundreds of megabytes
in the image and seconds of cold start, for a dictionary lookup.

The vocabulary is NOT hardcoded here. It is read from vocab.json, which the
training notebook writes from the real tokenizer - a remembered alphabet that
is subtly wrong would shift every token by one and produce confident nonsense,
which is the failure this whole service is supposed to be the opposite of.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np

# FASTA wraps sequences at 60 or 80 columns, so a pasted record arrives full of
# newlines. Treating those as residues would turn every line break into an
# unknown token and quietly degrade the answer, so whitespace is removed before
# anything else looks at the string.
WHITESPACE = re.compile(r"\s+")


def residues(sequence: str) -> str:
    return WHITESPACE.sub("", sequence).upper()

VOCAB_FILE = pathlib.Path(__file__).with_name("vocab.json")


class EsmTokenizer:
    def __init__(self, spec: dict):
        self.vocab: dict[str, int] = spec["vocab"]
        self.cls = spec["cls"]
        self.eos = spec["eos"]
        self.pad = spec["pad"]
        self.unk = spec["unk"]
        # Training truncated the string to max_len characters and then asked the
        # tokenizer for at most max_len tokens. Inference has to do the same or
        # long proteins are scored on a different slice than they were trained on.
        self.max_len: int = spec.get("max_len", 512)

    @classmethod
    def load(cls, path: pathlib.Path = VOCAB_FILE) -> EsmTokenizer:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def size(self) -> int:
        return len(self.vocab)

    def ids(self, sequence: str) -> list[int]:
        body = [self.vocab.get(ch, self.unk)
                for ch in residues(sequence)[: self.max_len]]
        out = [self.cls, *body, self.eos]
        return out[: self.max_len]

    def encode(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Pad to the longest in the batch, not to max_len.

        Padding everything to 512 would make most of the work happen on empty
        tokens: on a batch of short peptides that is an order of magnitude of
        wasted compute, and on Lambda compute is the bill.
        """
        if not sequences:
            raise ValueError("nothing to encode")
        rows = [self.ids(s) for s in sequences]
        width = max(len(r) for r in rows)
        input_ids = np.full((len(rows), width), self.pad, dtype=np.int64)
        attention = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = r
            attention[i, : len(r)] = 1
        return input_ids, attention


UNKNOWN_OK = set("XBUZO")


def unknown_residues(sequence: str, tok: EsmTokenizer) -> list[str]:
    """Characters the alphabet has no token for, reported rather than swallowed.

    X, B, U, Z and O are real ambiguity codes with their own tokens, so they are
    not flagged. Anything else - a digit, a space, a stray FASTA header - becomes
    <unk>, and the caller deserves to know that happened instead of receiving a
    number computed partly from padding.
    """
    seen = {ch for ch in residues(sequence) if ch not in tok.vocab}
    return sorted(seen - UNKNOWN_OK)
