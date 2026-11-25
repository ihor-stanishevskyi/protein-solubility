#!/usr/bin/env python3
"""Turn the notebook's metrics.json into the card that ships beside the weights.

Nothing here invents a number. Every figure on the card is copied from the
measurement the training run wrote down, and if a figure is missing the card
says the answers mean nothing rather than quietly leaving the field out.

The one trap this script exists to avoid: metrics.json calls the *hardware*
`trained_on` ("Colab T4, fp16 AMP"). The service prints the card's
`measured_on` as "MCC 0.47 on <this>", so copying that field across would put
"MCC 0.47 on Colab T4" in front of a reader. Hardware and dataset are kept
apart here deliberately.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REQUIRED = ("mcc", "auroc", "accuracy")


def build(metrics: dict, built: str) -> dict:
    missing = [k for k in REQUIRED if metrics.get(k) is None]
    base_model = metrics.get("model") or "an unnamed checkpoint"
    dataset = metrics.get("dataset") or "proteinea/solubility"
    n_test = metrics.get("n_test")
    n_train = metrics.get("n_train")

    card = {
        "kind": "trained",
        "name": f"{base_model} fine-tuned for solubility",
        "what_it_is": (
            f"{base_model} fine-tuned for {metrics.get('epochs', '?')} epochs on "
            f"the {dataset} training split"
            + (f" ({n_train} sequences)" if n_train else "")
            + f", exported to ONNX. Hardware: {metrics.get('trained_on', 'unrecorded')}."
        ),
        # The gate: a card without measurements is not a card for a real model.
        "answers_mean_nothing": bool(missing),
        "mcc": metrics.get("mcc"),
        "auroc": metrics.get("auroc"),
        "accuracy": metrics.get("accuracy"),
        # When the run did not write down its own date, the card says so.
        # Filling in today's date would claim a measurement was made today.
        "measured": metrics.get("measured_at"),
        "card_built": built,
        "trained_on": f"{dataset} training split"
                      + (f", {n_train} sequences" if n_train else ""),
        "measured_on": f"the {dataset} test split"
                       + (f", {n_test} sequences" if n_test else ""),
        "scope": (
            "The training set is bacterial expression data, so a prediction for "
            "a protein from another organism or expression system is extrapolation."
        ),
    }

    if missing:
        # The name is printed as "running a {name}, not a trained model", so it
        # cannot go on calling itself a fine-tuned solubility model.
        card["name"] = "model with unverified performance"
        card["what_it_is"] = (
            f"A model whose measurements are incomplete: {', '.join(missing)} "
            f"missing from metrics.json. Nothing can be claimed about it."
        )

    # A run that did not reproduce the reference is not silently blessed.
    ref = metrics.get("reference_2026_11_07") or {}
    if ref.get("mcc") is not None and card["mcc"] is not None:
        drift = abs(card["mcc"] - ref["mcc"])
        card["reproduces_reference"] = drift <= 0.05
        card["reference_mcc"] = ref["mcc"]
        if drift > 0.05:
            card["scope"] = (
                f"This run scored MCC {card['mcc']}, against {ref['mcc']} measured "
                f"on 2026-11-07 - a gap wider than run-to-run noise, and unexplained. "
                + card["scope"]
            )

    diff = metrics.get("onnx_vs_torch_max_diff")
    if diff is not None:
        card["onnx_matches_torch_to"] = diff

    return card


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metrics", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--measured", required=True, dest="built",
                    help="today's date; used only as the card's build date")
    a = ap.parse_args()

    metrics = json.loads(a.metrics.read_text(encoding="utf-8"))
    card = build(metrics, a.built)
    a.out.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if card["answers_mean_nothing"]:
        print(f"WARNING: {a.out} says the answers mean nothing - metrics.json is "
              f"missing {[k for k in REQUIRED if metrics.get(k) is None]}", file=sys.stderr)
    else:
        print(f"{a.out}: MCC {card['mcc']} on {card['measured_on']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
