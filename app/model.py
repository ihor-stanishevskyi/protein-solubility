"""Loading the model, and refusing to serve one the vocabulary does not fit.

The danger this guards against is specific. If vocab.json and the model come
from different runs, every residue maps to the wrong token and the service
returns confident, plausible, meaningless probabilities - the exact failure
mode of the library this project's sibling exists to catch. So the embedding's
first dimension is compared with the alphabet, and a mismatch is fatal.

Where the check cannot be made, the service says so in /health instead of
implying it passed.
"""
from __future__ import annotations

import functools
import os
import pathlib

import numpy as np
import onnxruntime as ort

from .tokenizer import EsmTokenizer

MODEL_FILE = pathlib.Path(os.environ.get(
    "MODEL_PATH", pathlib.Path(__file__).with_name("model.onnx")))

EMBEDDING_HINTS = ("word_embeddings", "embed_tokens", "embeddings.weight")


class VocabMismatch(RuntimeError):
    pass


def _declared_vocab(model_file: pathlib.Path) -> tuple[int | None, str]:
    """The model's own idea of how many tokens exist, if it can be found.

    Takes the path being served, not the module default. The first version read
    MODEL_FILE while the session ran a different file, so the guard inspected
    one model and cleared another - a check that cannot fail is worse than no
    check, because it reads like protection. A test caught it.
    """
    try:
        import onnx
    except ImportError:
        return None, "onnx not installed in the runtime image, by design"
    try:
        g = onnx.load(str(model_file), load_external_data=False).graph
    except Exception as e:  # noqa: BLE001 - onnx raises many types; the point is
        # that the failure is reported out through /health rather than swallowed,
        # so the reader knows the check did not happen.
        return None, f"graph could not be read: {type(e).__name__}"
    named = [t for t in g.initializer
             if any(h in t.name for h in EMBEDDING_HINTS) and len(t.dims) == 2]
    if named:
        return named[0].dims[0], f"from {named[0].name}"
    small = [t for t in g.initializer if len(t.dims) == 2 and t.dims[0] < 1000]
    if len(small) == 1:
        return small[0].dims[0], f"from {small[0].name} (only 2-D candidate)"
    return None, "no embedding matrix identified in the graph"


class Predictor:
    def __init__(self, model_file: pathlib.Path = MODEL_FILE):
        if not model_file.exists():
            raise FileNotFoundError(
                f"no model at {model_file}. The image is built with one baked in; "
                f"set MODEL_PATH to run against another.")
        self.path = model_file
        self.tok = EsmTokenizer.load()
        opts = ort.SessionOptions()
        # One thread: Lambda gives one core per 1769 MB, and onnxruntime
        # spinning up a pool it cannot use costs cold-start time.
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        self.sess = ort.InferenceSession(str(model_file), opts,
                                         providers=["CPUExecutionProvider"])
        declared, how = _declared_vocab(model_file)
        self.vocab_check = how
        if declared is not None and declared != self.tok.size:
            raise VocabMismatch(
                f"the model expects {declared} tokens, the alphabet in vocab.json "
                f"has {self.tok.size}. Every residue would map to the wrong token "
                f"and the answers would look fine and mean nothing. Refusing.")
        self.vocab_verified = declared is not None

    def predict(self, sequences: list[str]) -> np.ndarray:
        ids, att = self.tok.encode(sequences)
        logits = self.sess.run(None, {"input_ids": ids, "attention_mask": att})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(axis=1, keepdims=True))[:, 1]


@functools.lru_cache(maxsize=1)
def predictor() -> Predictor:
    return Predictor()
