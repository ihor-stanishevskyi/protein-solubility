"""Loading the model, and refusing to serve one the alphabet does not fit.

The danger is specific. If vocab.json and model.onnx come from different runs,
every residue maps to the wrong token and the service returns confident,
plausible, meaningless probabilities - the exact failure this project's sibling
exists to catch in someone else's library.

How the check is made matters as much as that it exists. The first version read
the model's embedding matrix with the `onnx` library, which is a build-time
dependency and deliberately absent from the runtime image - so the guard passed
in tests and did nothing in production, which CI caught by asking the running
container whether the check had happened.

So it is done by asking the model instead of reading it. onnxruntime enforces
embedding bounds, so the highest token id the model accepts IS its vocabulary
size. Same code in both environments, no extra dependency, and it tests
behaviour rather than metadata.

What it catches: an alphabet of the wrong size. What it does not: an alphabet
of the right size in the wrong order. /health says so rather than implying the
model has been fully vouched for.
"""
from __future__ import annotations

import functools
import json
import os
import pathlib

import numpy as np
import onnxruntime as ort

from .tokenizer import EsmTokenizer

MODEL_FILE = pathlib.Path(os.environ.get(
    "MODEL_PATH", pathlib.Path(__file__).with_name("model.onnx")))

# The measured numbers belong to a particular set of weights, so they ship
# beside those weights rather than living in the code. Hardcoding them meant a
# stand-in model would have been served under the real model's MCC - the
# service confidently quoting a score it had not earned, which is the exact
# failure this project's sibling exists to catch.
UNKNOWN_CARD = {
    "kind": "unknown",
    "name": "unidentified model",
    "what_it_is": "No card shipped beside this model file, so nothing is known "
                  "about how it was trained or how well it does.",
    "answers_mean_nothing": True,
    "mcc": None, "auroc": None, "accuracy": None, "measured": None,
    "trained_on": None, "measured_on": None, "scope": None,
}


def load_card(model_file: pathlib.Path) -> dict:
    card = model_file.with_suffix(".card.json")
    if not card.exists():
        return dict(UNKNOWN_CARD)
    try:
        return json.loads(card.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return dict(UNKNOWN_CARD) | {
            "what_it_is": f"The card beside this model is not readable ({e}), so "
                          f"nothing is known about it."}

CHECK_NOTE = ("the highest token id the model accepts matches the alphabet; this "
              "catches an alphabet of the wrong size, not one of the right size "
              "in the wrong order")


class VocabMismatch(RuntimeError):
    pass


def _accepts(sess: ort.InferenceSession, token_id: int) -> bool:
    ids = np.array([[token_id]], dtype=np.int64)
    try:
        sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    except Exception:  # noqa: BLE001 - any refusal means out of range
        return False
    return True


def _model_vocab(sess: ort.InferenceSession, limit: int = 1 << 20) -> int:
    """Largest accepted id plus one, by bisection. Only called when something
    already looks wrong, so the extra forward passes cost nothing in the
    ordinary case."""
    if not _accepts(sess, 0):
        return 0
    lo, hi = 0, 1
    while hi < limit and _accepts(sess, hi):
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _accepts(sess, mid):
            lo = mid
        else:
            hi = mid
    return lo + 1


class Predictor:
    def __init__(self, model_file: pathlib.Path = MODEL_FILE):
        if not model_file.exists():
            raise FileNotFoundError(
                f"no model at {model_file}. The image is built with one baked in; "
                f"set MODEL_PATH to run against another.")
        self.path = model_file
        self.tok = EsmTokenizer.load()
        opts = ort.SessionOptions()
        # One thread: Lambda gives one core per 1769 MB, and a pool it cannot
        # use costs cold-start time.
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        # The probe below deliberately feeds an out-of-range id, and
        # onnxruntime logs that at error level. It is expected, so it is not
        # printed - a scary log line for a check that is working teaches people
        # to ignore the logs.
        opts.log_severity_level = 4
        self.sess = ort.InferenceSession(str(model_file), opts,
                                         providers=["CPUExecutionProvider"])

        size = self.tok.size
        fits = _accepts(self.sess, size - 1) and not _accepts(self.sess, size)
        if not fits:
            found = _model_vocab(self.sess)
            raise VocabMismatch(
                f"the model accepts {found} tokens, the alphabet in vocab.json has "
                f"{size}. Every residue would map to the wrong token and the answers "
                f"would look fine and mean nothing. Refusing to serve.")
        self.vocab_verified = True
        self.vocab_check = CHECK_NOTE
        self.card = load_card(model_file)

    def predict(self, sequences: list[str]) -> np.ndarray:
        ids, att = self.tok.encode(sequences)
        logits = self.sess.run(None, {"input_ids": ids, "attention_mask": att})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(axis=1, keepdims=True))[:, 1]


@functools.lru_cache(maxsize=1)
def predictor() -> Predictor:
    return Predictor()
