"""The ESM-2 alphabet as a fixture, so tests and the local run need no download.

This is NOT the source of truth. The real vocab.json comes out of the training
notebook, written from the tokenizer that actually produced the model. This copy
exists so the service can be built and tested before the model exists, and
app/model.py refuses to serve if the two ever disagree about the vocabulary size.
"""
import json
import pathlib

# Standard ESM-2 alphabet: 4 special tokens, then residues in the order the
# model was trained with.
TOKENS = ["<cls>", "<pad>", "<eos>", "<unk>",
          "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
          "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
          "<null_1>", "<mask>"]

spec = {
    "vocab": {t: i for i, t in enumerate(TOKENS)},
    "cls": 0, "eos": 2, "pad": 1, "unk": 3,
    "max_len": 512,
    "note": "fixture written by scripts/make_test_vocab.py; the real one comes "
            "from the training notebook",
}
out = pathlib.Path(__file__).resolve().parent.parent / "app" / "vocab.json"
out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
print(f"{out}: {len(TOKENS)} tokens")
