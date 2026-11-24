"""Protein solubility prediction, served with its own limits attached.

The model is a fine-tuned ESM-2 35M. What it scored, and on what, is read from
the card that ships beside the weights - never written into this code, and never
into the page. A bare number between 0 and 1 invites more confidence than any of
these models has earned, so every answer carries the card's own caveat.
"""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .model import VocabMismatch, predictor
from .page import PAGE

__version__ = "0.1.0"

# The score AND the set it was measured on both come from the card. Naming a
# dataset here would be the same defect as naming a score here: the sentence
# outlives the weights it was written for.
REAL_CAVEAT = (
    "MCC {mcc} on {measured_on}: {standing}, and far from a decision you should "
    "act on for one protein. {scope} This is not a laboratory result and not "
    "advice."
)


def standing(mcc: float | None) -> str:
    """Describe the score the weights actually got, not the one we hoped for.

    A fixed phrase here would praise a bad model in the same words as a good
    one - the template asserting what only the number can say.
    """
    if mcc is None:
        return "no score was recorded"
    if mcc >= 0.3:
        return "better than chance by a wide margin"
    if mcc >= 0.1:
        return "better than chance, but not by much"
    return "barely distinguishable from chance"
DEFAULT_SCOPE = (
    "A prediction for a protein unlike those it was trained on is extrapolation."
)
HOLLOW_CAVEAT = (
    "THESE NUMBERS MEAN NOTHING. This service is running a {name}, not a trained "
    "model: {what}. Every probability below is arbitrary. The service is live so "
    "that the deployment can be tested before the model exists."
)


def caveat(card: dict) -> str:
    """Never quote a score the loaded weights did not earn."""
    if card.get("answers_mean_nothing"):
        what = (card.get("what_it_is") or "").strip().rstrip(".")
        return HOLLOW_CAVEAT.format(name=card.get("name", "placeholder"), what=what)
    return REAL_CAVEAT.format(
        mcc=card.get("mcc"),
        standing=standing(card.get("mcc")),
        measured_on=card.get("measured_on") or "the set it was evaluated on",
        scope=(card.get("scope") or DEFAULT_SCOPE).strip(),
    )

MAX_SEQUENCES = 64
MAX_RESIDUES = 64 * 512

app = FastAPI(
    title="Protein solubility",
    description="Fine-tuned ESM-2 35M, served on CPU. The measured performance "
                "and its caveat travel with whichever weights are loaded; see /health.",
    version=__version__,
)


class PredictRequest(BaseModel):
    sequences: list[str] = Field(..., min_length=1, max_length=MAX_SEQUENCES,
                                 description="Amino-acid sequences, one-letter code.")
    names: list[str] | None = Field(None, description="Optional labels, same order.")


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE)


@app.get("/health", tags=["ops"])
def health():
    try:
        p = predictor()
    except (FileNotFoundError, VocabMismatch) as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}") from e
    return {
        "status": "ok",
        "version": __version__,
        "model_file": p.path.name,
        "alphabet_size": p.tok.size,
        "max_residues_scored": p.tok.max_len,
        # Whether the alphabet was actually checked against the model, not
        # whether we hope it matches.
        "vocabulary_checked_against_model": p.vocab_verified,
        "vocabulary_check": p.vocab_check,
        "model": p.card,
        "caveat": caveat(p.card),
    }


@app.post("/predict", tags=["predict"])
def predict(req: PredictRequest):
    from .tokenizer import unknown_residues

    seqs = [s.strip() for s in req.sequences]
    if any(not s for s in seqs):
        raise HTTPException(400, "one of the sequences is empty")
    total = sum(len(s) for s in seqs)
    if total > MAX_RESIDUES:
        raise HTTPException(413, f"{total} residues in one request; the limit is "
                                 f"{MAX_RESIDUES}. Split them.")
    if req.names is not None and len(req.names) != len(seqs):
        raise HTTPException(400, f"{len(req.names)} names for {len(seqs)} sequences")

    try:
        p = predictor()
    except (FileNotFoundError, VocabMismatch) as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}") from e

    probs = p.predict(seqs)
    results = []
    for i, (s, prob) in enumerate(zip(seqs, probs, strict=True)):
        odd = unknown_residues(s, p.tok)
        results.append({
            "name": req.names[i] if req.names else f"sequence_{i + 1}",
            "length": len(s),
            # Truncation is reported, never silent: a protein scored on its
            # first 510 residues was not scored on the protein.
            "scored_residues": min(len(s), p.tok.max_len - 2),
            "truncated": len(s) > p.tok.max_len - 2,
            "probability_soluble": round(float(prob), 4),
            "call": "soluble" if prob > 0.5 else "insoluble",
            "unrecognised_characters": odd or None,
        })
    return {"results": results, "model": p.card, "caveat": caveat(p.card)}


@app.get("/example", tags=["predict"])
def example(
    which: Annotated[str, Query(description="green_fluorescent_protein or "
                                           "inclusion_body_prone")] = "green_fluorescent_protein",
):
    """Two sequences to try, so the page has something to show immediately."""
    if which not in EXAMPLES:
        raise HTTPException(404, f"unknown example {which!r}; have {sorted(EXAMPLES)}")
    return {"name": which, **EXAMPLES[which]}


EXAMPLES = {
    "green_fluorescent_protein": {
        "sequence": (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTF"
            "SYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGI"
            "DFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGP"
            "VLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
        "note": "GFP, the classic well-behaved soluble protein.",
    },
    "inclusion_body_prone": {
        "sequence": (
            "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFGFFFFLLLLWWWWIIIIVVVVAAAA"
            "FFFFLLLLWWWWIIIIVVVVAAAAFFFFLLLLWWWWIIIIVVVV"),
        "note": "A hydrophobic stretch of the kind that tends to aggregate. "
                "Invented for illustration, not a real protein.",
    },
}

if os.environ.get("CORS_ALLOW_ORIGINS"):
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ["CORS_ALLOW_ORIGINS"].split(",")],
        allow_methods=["GET", "POST"], allow_headers=["content-type"],
    )
