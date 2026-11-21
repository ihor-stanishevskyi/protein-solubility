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
    assert "word_embeddings" in b["vocabulary_check"]


def test_every_answer_carries_the_model_s_measured_limits():
    """A bare number between 0 and 1 invites more confidence than MCC 0.475
    has earned, so the figure travels with the answer, not only the README."""
    for body in (client.get("/health").json(),
                 client.post("/predict", json={"sequences": [GFP]}).json()):
        assert body["performance"]["mcc"] == 0.4754
        assert "not advice" in body["caveat"]


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


def test_a_mismatched_alphabet_stops_the_service_instead_of_answering(tmp_path):
    """The failure this guards against: vocab.json and the model from different
    runs means every residue maps to the wrong token, and the answers look
    exactly as plausible as correct ones."""
    import onnx
    from onnx import numpy_helper

    src = pathlib.Path(__file__).resolve().parent.parent / "models" / "stand-in.onnx"
    m = onnx.load(str(src))
    for t in m.graph.initializer:
        if "word_embeddings" in t.name:
            a = numpy_helper.to_array(t)
            t.CopyFrom(numpy_helper.from_array(
                np.vstack([a, a[:5]]).astype(np.float32), t.name))   # 33 -> 38
    bad = tmp_path / "wrong_vocab.onnx"
    onnx.save(m, bad)

    with pytest.raises(VocabMismatch) as e:
        Predictor(bad)
    assert "38" in str(e.value) and "33" in str(e.value)


def test_a_missing_model_is_a_clear_failure_not_a_stack_trace(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        Predictor(tmp_path / "absent.onnx")
    assert "MODEL_PATH" in str(e.value)
