# protein-solubility

Will this protein express solubly in *E. coli*? A fine-tuned ESM-2 (35M) gives a
probability, served on AWS Lambda from a container, with no PyTorch anywhere near
the runtime.

**[Try it](https://lcfxj6aqrznnjwfe7agp5j3pam0yhtva.lambda-url.us-east-1.on.aws/)**

## What the number is worth

The same recipe was trained three times, deliberately unchanged, so that the
spread is visible rather than hidden behind one figure:

| Run | GPU | MCC | AUROC | Accuracy |
|---|---|---|---|---|
| 2026-11-07 | RTX 4090 | 0.4754 | 0.8344 | 73.6% |
| **2026-11-24** (deployed) | Tesla T4 | **0.4594** | **0.8319** | **72.0%** |
| 2026-11-25 | Tesla T4 | 0.4336 | 0.8352 | 70.1% |

**AUROC barely moves — 0.832 to 0.835 — while MCC swings by 0.04.** MCC is
computed at a fixed 0.5 threshold, so it picks up every small shift in where
the probabilities land; AUROC does not depend on a threshold at all. So the
honest reading of the deployed number is not "MCC 0.4594" but "MCC somewhere
around 0.43–0.48, and a ranking quality that is stable".

The service reports the measured figures of the weights it is actually
running, which is the middle row. `/health` always states them.

0.46 is clearly better than chance and clearly not a decision. It is the
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

So on startup the service asks the model directly: onnxruntime enforces
embedding bounds, so the highest token id the model accepts *is* its vocabulary
size. A mismatch stops the service rather than letting it answer.

The first version read the embedding matrix with the `onnx` library instead -
which is a build dependency, absent from the runtime image. The guard passed in
tests and did nothing in production. CI caught it by asking the running
container whether the check had happened; a test now fails if anything in `app/`
imports something `requirements.txt` does not install.

`/health` reports whether the check ran, and what it does not cover: it catches
an alphabet of the wrong size, not one of the right size in the wrong order.

## Infrastructure

`infra/roles.yaml` creates both IAM roles: the one GitHub Actions assumes to
deploy, and the one the function runs as. No access keys exist anywhere; CI
authenticates with a short-lived OpenID Connect token.

One detail worth copying. GitHub now puts numeric owner and repository ids in
the token subject:

```
repo:owner@311364649/protein-solubility@1465118856:ref:refs/heads/main
```

The repository id cannot be known before the repository exists, and the AWS
console wizard still writes the older format without it. The trust policy
wildcards *only* that number and keeps owner, owner id, repository name and
branch exact, which removes the create-fail-read-the-log-fix loop entirely.

## Running it

```bash
pip install -r requirements-dev.txt
pytest -q
uvicorn app.main:app --reload
```

The repository ships a stand-in model (`models/stand-in.onnx`, 1.7 KB) with the
real one's input and output signature. Its answers are meaningless on purpose:
tests that needed the trained weights would be testing the model, not the
service. The image bakes that stand-in in by default; the deploy workflow
replaces it with the trained weights from the project's S3 bucket, and says in
its job summary which of the two it shipped.

Nothing the service claims about a model is written in the code. The score, the
set it was measured on, the alphabet and the wording of the caveat all come from
a card built by `scripts/build_card.py` out of the training run's own
`metrics.json`, and travel in the image beside the weights they describe. A run
that lost one of its measurements produces a card that says the answers mean
nothing; a run that drifts from the reference says so in the caveat. The reason
is in the git history: the page spent an afternoon claiming the real model's
accuracy above the random-weights stand-in, because that sentence had been typed
into the HTML while the plan was still that the real weights would be there.

## Endpoints

| | |
|---|---|
| `GET /` | the page a person opens |
| `GET /health` | status, alphabet size, whether the vocabulary check ran, and the card of the weights actually loaded |
| `POST /predict` | `{"sequences": [...]}` → probability per sequence |
| `GET /example` | two sequences to try |

Every answer carries the measured performance and the caveat. Truncation beyond
512 residues and characters the alphabet does not know are reported per
sequence, never applied silently.

## Not a laboratory result

Trained on bacterial expression data. A prediction for a protein from another
organism or expression system is extrapolation. This is not advice.
