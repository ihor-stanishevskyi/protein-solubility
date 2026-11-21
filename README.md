# protein-solubility

Will this protein express solubly in *E. coli*? A fine-tuned ESM-2 (35M) gives a
probability, served on AWS Lambda from a container, with no PyTorch anywhere near
the runtime.

**[Try it](#)** · the live URL goes here once the first deploy lands.

## What the number is worth

| | |
|---|---|
| MCC | **0.475** |
| AUROC | 0.834 |
| Accuracy | 73.6% |
| Measured on | DeepSol test split, 2001 proteins, 2026-11-07 |

0.475 is clearly better than chance and clearly not a decision. It is the
difference between *"rank these 200 constructs and try the top 20 first"* and
*"do not bother expressing this one"* — only the first is supported.

For context, published linear probing on the same dataset reports MCC 0.342 for
frozen ESM-2 35M, 0.423 for ESM-2 **3B** and 0.432 for ProGen2 Large. Fine-tuning
the small model beats freezing the large one, which is the interesting part: how
a model is used mattered more here than how big it was.

## Why there is no PyTorch in the image

| | wheel |
|---|---|
| `torch` 2.15.0 | 539 MB |
| `onnxruntime` + `numpy` | 44 MB |

The model is exported to ONNX after training. That turns a ~2 GB image into a
~250 MB one, which on Lambda is the difference between a slow cold start and a
fast one. ESM-2's alphabet is 33 entries, so the tokenizer is written out here
too and `transformers` is not needed at runtime either.

## The guard that matters

`vocab.json` and `model.onnx` must come from the same training run. If they do
not, every residue maps to the wrong token and the service returns confident,
plausible, meaningless probabilities — the failure this project's
[sibling](https://github.com/ihor-stanishevskyi/clock-audit) exists to catch in
someone else's library.

So on startup the alphabet is compared against the model's own embedding matrix,
and a mismatch stops the service rather than letting it answer. `/health` reports
whether that comparison actually happened, not whether we hope it passed.

## Running it

```bash
pip install -r requirements-dev.txt
pytest -q
uvicorn app.main:app --reload
```

The repository ships a stand-in model (`models/stand-in.onnx`, 1.7 KB) with the
real one's input and output signature. Its answers are meaningless on purpose:
tests that needed the trained weights would be testing the model, not the
service. The image bakes that stand-in in by default and CI overrides
`MODEL_SOURCE` with the real one.

## Endpoints

| | |
|---|---|
| `GET /` | the page a person opens |
| `GET /health` | status, alphabet size, whether the vocabulary check ran, measured performance |
| `POST /predict` | `{"sequences": [...]}` → probability per sequence |
| `GET /example` | two sequences to try |

Every answer carries the measured performance and the caveat. Truncation beyond
512 residues and characters the alphabet does not know are reported per
sequence, never applied silently.

## Not a laboratory result

Trained on *E. coli* expression data. A prediction for another organism or
expression system is extrapolation. This is not advice.
