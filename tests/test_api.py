import pathlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import MAX_RESIDUES, app
from app.model import Predictor, VocabMismatch

client = TestClient(app)
GFP = ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTF"
       "SYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGI")


def test_health_reports_what_was_actually_checked():
    b = client.get("/health").json()
    assert b["status"] == "ok"
    assert b["alphabet_size"] == 33
    # Not "we hope it matches" - whether the comparison was made.
    assert b["vocabulary_checked_against_model"] is True
    # And it says what the check does NOT cover, rather than implying the
    # model has been fully vouched for.
    assert "not one of the right size in the wrong order" in b["vocabulary_check"]


def test_a_stand_in_model_is_never_served_under_the_real_one_s_score():
    """Nearly shipped: the measured numbers were written into the code, so the
    placeholder would have answered with random probabilities under MCC 0.475 -
    a service confidently quoting a score its weights had not earned.

    The card now travels with the weights. Running the stand-in, the service
    has to say so, loudly, in every answer."""
    for body in (client.get("/health").json(),
                 client.post("/predict", json={"sequences": [GFP]}).json()):
        assert body["model"]["kind"] == "stand-in"
        assert body["model"]["mcc"] is None
        assert "MEAN NOTHING" in body["caveat"]
        assert "0.475" not in body["caveat"]


def test_a_model_with_no_card_is_treated_as_unknown_not_as_good(tmp_path):
    """Absence of a claim must not read as a good claim."""
    import shutil

    src = pathlib.Path(__file__).resolve().parent.parent / "models" / "stand-in.onnx"
    lonely = tmp_path / "nameless.onnx"
    shutil.copy(src, lonely)
    p = Predictor(lonely)
    assert p.card["kind"] == "unknown"
    assert p.card["answers_mean_nothing"] is True


def test_a_real_card_restores_the_real_caveat(tmp_path):
    import json as j
    import shutil

    src = pathlib.Path(__file__).resolve().parent.parent / "models" / "stand-in.onnx"
    m = tmp_path / "pretend-real.onnx"
    shutil.copy(src, m)
    m.with_suffix(".card.json").write_text(j.dumps({
        "kind": "trained", "name": "fine-tuned ESM-2 35M", "answers_mean_nothing": False,
        "mcc": 0.4754, "auroc": 0.8344, "accuracy": 0.7361,
        "trained_on": "DeepSol", "measured": "2026-11-07"}))
    from app.main import caveat
    c = caveat(Predictor(m).card)
    assert "0.4754" in c and "not advice" in c and "MEAN NOTHING" not in c


def test_predict_returns_a_probability_and_a_call_that_agree():
    r = client.post("/predict", json={"sequences": [GFP]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert 0.0 <= res["probability_soluble"] <= 1.0
    assert res["call"] == ("soluble" if res["probability_soluble"] > 0.5 else "insoluble")
    assert res["length"] == len(GFP)
    assert res["truncated"] is False


def test_a_batch_keeps_its_order_and_names():
    r = client.post("/predict", json={"sequences": [GFP, "MKTV", GFP[:40]],
                                      "names": ["a", "b", "c"]}).json()
    assert [x["name"] for x in r["results"]] == ["a", "b", "c"]
    assert [x["length"] for x in r["results"]] == [len(GFP), 4, 40]


def test_padding_does_not_change_a_sequence_s_own_answer():
    """A short peptide batched beside a long protein gets padded. If the mask
    were wrong, its answer would drift with its neighbours - and nothing else
    would report that."""
    one = client.post("/predict", json={"sequences": ["MKTV"]})
    alone = one.json()["results"][0]
    pair = client.post("/predict", json={"sequences": [GFP, "MKTV"]})
    crowded = pair.json()["results"][1]
    assert abs(alone["probability_soluble"] - crowded["probability_soluble"]) < 1e-5


def test_truncation_is_reported_never_silent():
    long = "A" * 900
    res = client.post("/predict", json={"sequences": [long]}).json()["results"][0]
    assert res["truncated"] is True
    assert res["scored_residues"] == 510 and res["length"] == 900


def test_unrecognised_characters_are_named():
    res = client.post("/predict", json={"sequences": ["MKTV123"]}).json()["results"][0]
    assert res["unrecognised_characters"] == ["1", "2", "3"]


def test_a_wrapped_fasta_sequence_scores_as_the_flat_one():
    a = client.post("/predict", json={"sequences": [GFP]}).json()["results"][0]
    wrapped = "\n".join(GFP[i:i + 60]
                        for i in range(0, len(GFP), 60))
    b = client.post("/predict", json={"sequences": [wrapped]}).json()["results"][0]
    assert a["probability_soluble"] == b["probability_soluble"]
    assert b["unrecognised_characters"] is None


def test_oversized_and_malformed_requests_are_refused_with_a_reason():
    assert client.post("/predict", json={"sequences": []}).status_code == 422
    assert client.post("/predict", json={"sequences": ["  "]}).status_code == 400
    mismatched = client.post("/predict",
                             json={"sequences": [GFP], "names": ["a", "b"]})
    assert mismatched.status_code == 400
    big = client.post("/predict", json={"sequences": ["A" * (MAX_RESIDUES + 1)]})
    assert big.status_code == 413 and str(MAX_RESIDUES) in big.json()["detail"]


def test_examples_exist_and_can_be_scored():
    e = client.get("/example").json()
    assert e["sequence"].startswith("MSKGEELF")
    assert client.post("/predict", json={"sequences": [e["sequence"]]}).status_code == 200
    assert client.get("/example", params={"which": "nope"}).status_code == 404


def test_index_is_a_self_contained_page():
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    # A Lambda container has no business fetching a CDN, and the page should
    # render for someone offline of everything but this service.
    assert "http://" not in r.text and "https://" not in r.text


@pytest.mark.parametrize(("rows", "shown"), [(38, "38"), (20, "20")])
def test_a_mismatched_alphabet_stops_the_service_instead_of_answering(tmp_path, rows, shown):
    """The failure this guards against: vocab.json and the model from different
    runs means every residue maps to the wrong token, and the answers look
    exactly as plausible as correct ones.

    Both directions are tested. The first version of the guard only ever saw
    the file it was configured with rather than the one being served, so it
    could not fail at all - and a check that cannot fail reads like protection
    while providing none."""
    import onnx
    from onnx import numpy_helper

    src = pathlib.Path(__file__).resolve().parent.parent / "models" / "stand-in.onnx"
    m = onnx.load(str(src))
    for t in m.graph.initializer:
        if "word_embeddings" in t.name:
            a = numpy_helper.to_array(t)
            grown = np.vstack([a] * 3)[:rows] if rows > len(a) else a[:rows]
            t.CopyFrom(numpy_helper.from_array(grown.astype(np.float32), t.name))
    bad = tmp_path / f"wrong_vocab_{rows}.onnx"
    onnx.save(m, bad)

    with pytest.raises(VocabMismatch) as e:
        Predictor(bad)
    assert shown in str(e.value) and "33" in str(e.value)


def test_the_guard_needs_nothing_the_runtime_image_lacks():
    """It failed in production once because it used `onnx`, which is a build
    dependency and absent from the image. Whatever it imports now has to be in
    the runtime requirements, or dev passes while production does nothing."""
    import pathlib as pl
    import sys

    here = pl.Path(__file__).resolve().parent.parent
    runtime = (here / "requirements.txt").read_text()
    for name in ("model", "tokenizer", "main", "page"):
        for line in (here / "app" / f"{name}.py").read_text().splitlines():
            if not (line.startswith("import ") or line.startswith("from ")):
                continue
            mod = line.split()[1].split(".")[0]
            # Relative imports and the standard library ship with Python.
            if not mod or mod.startswith(".") or mod in sys.stdlib_module_names:
                continue
            assert mod in runtime, (
                f"app/{name}.py imports {mod!r}, which requirements.txt does not "
                f"install - it would work in tests and be missing in the image")


def test_a_missing_model_is_a_clear_failure_not_a_stack_trace(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        Predictor(tmp_path / "absent.onnx")
    assert "MODEL_PATH" in str(e.value)


# --- the page itself ---------------------------------------------------
#
# Three defects shipped to production at once here, and no test saw any of
# them: the HTML claimed the real model's accuracy above a random-weights
# model, the script read a /health field that had been renamed, and the
# crash from that rename replaced the honest caveat with "could not reach
# the service" - while the service was answering perfectly. CI had asserted
# the JSON and never looked at the page.

def _page_parts():
    from app.page import PAGE
    prose = PAGE.split("</style>", 1)[1].split("<script>", 1)[0]
    script = PAGE.split("<script>", 1)[1]
    return prose, script


def test_the_page_quotes_no_score_of_its_own():
    """Prose outlives the weights it described, and then it is a lie."""
    import re
    prose, _ = _page_parts()
    bad = re.findall(r"MCC|AUROC|\d+(?:\.\d+)?\s*%|times in (?:four|five|ten)|"
                     r"accurate|accuracy", prose, re.I)
    assert not bad, f"the HTML makes its own performance claim: {bad}"


def test_the_page_reads_only_fields_health_actually_returns():
    """It read h.performance for a day after /health renamed it h.model."""
    import re
    _, script = _page_parts()
    body = client.get("/health").json()

    used = set(re.findall(r"\bh\.([a-z_]+)", script))
    assert used, "the script reads nothing from /health - has it changed shape?"
    assert used <= set(body), f"page reads {sorted(used - set(body))}; /health has no such field"

    used_card = set(re.findall(r"\bm\.([a-z_]+)", script))
    assert used_card, "the script reads nothing from the model card"
    assert used_card <= set(body["model"]), (
        f"page reads {sorted(used_card - set(body['model']))} off the card; it has no such field")


def test_the_page_can_say_the_model_is_hollow():
    """The red banner existed in the CSS and nothing ever filled it."""
    prose, script = _page_parts()
    assert 'id="alarm"' in prose and 'id="claim"' in prose
    assert "answers_mean_nothing" in script, "nothing on the page reacts to a stand-in"
    assert "$(\"#alarm\").className" in script, "the alarm element is never given its style"


def test_the_caveat_reads_as_one_sentence():
    b = client.get("/health").json()
    assert ".." not in b["caveat"]
