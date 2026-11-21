import json
import pathlib

import numpy as np
import pytest

from app.tokenizer import EsmTokenizer, unknown_residues

tok = EsmTokenizer.load()


def test_special_tokens_wrap_the_sequence():
    ids = tok.ids("MKTV")
    assert ids[0] == tok.cls and ids[-1] == tok.eos
    assert len(ids) == 6


def test_padding_is_to_the_longest_in_the_batch_not_to_the_maximum():
    """Padding every batch to 512 would spend most of the compute on empty
    tokens, and on Lambda compute is the bill."""
    ids, att = tok.encode(["MKTV", "MKTVRQERLK"])
    assert ids.shape == (2, 12)
    assert att[0].sum() == 6 and att[1].sum() == 12
    assert (ids[0, 6:] == tok.pad).all()


def test_masked_positions_are_padding_and_nothing_else():
    ids, att = tok.encode(["MKTV", "M"])
    assert set(ids[att == 0].tolist()) == {tok.pad}


def test_a_long_protein_is_cut_at_the_trained_limit():
    """Training truncated at 512. Scoring a longer protein on more than that
    would be scoring it differently than the model was taught."""
    assert len(tok.ids("A" * 5000)) == tok.max_len


def test_lowercase_is_accepted():
    assert tok.ids("mktv") == tok.ids("MKTV")


def test_ambiguity_codes_are_not_flagged_but_junk_is():
    """X, B, U, Z and O are real residue codes with their own tokens. A digit
    is not, and silently turning it into <unk> would hide a broken input."""
    assert unknown_residues("MKTVXBUZO", tok) == []
    assert unknown_residues("MKTV-123*", tok) == ["*", "1", "2", "3"]


def test_a_wrapped_fasta_sequence_is_not_full_of_unknown_tokens():
    """Found by a test, not in production: FASTA wraps at 60 columns, so a
    pasted record is mostly newlines. Counting those as residues would make
    every line break an unknown token and weaken the answer silently."""
    wrapped = "MKTVRQERLK\nSIVRILERSK\n EPVSGAQ\t"
    flat = "MKTVRQERLKSIVRILERSKEPVSGAQ"
    assert unknown_residues(wrapped, tok) == []
    assert tok.ids(wrapped) == tok.ids(flat)


def test_unknown_characters_map_to_unk_rather_than_crashing():
    ids = tok.ids("MK1V")
    assert tok.unk in ids


def test_the_alphabet_is_read_from_file_not_hardcoded():
    """A remembered alphabet that is subtly wrong shifts every token and
    produces confident nonsense, so the file is the only source."""
    here = pathlib.Path(tok.__module__.replace(".", "/")).parent
    spec = json.loads((here / "vocab.json").read_text())
    assert len(spec["vocab"]) == tok.size == 33
    assert spec["vocab"]["<pad>"] == tok.pad


def test_encode_returns_int64_which_is_what_onnx_expects():
    ids, att = tok.encode(["MKTV"])
    assert ids.dtype == np.int64 and att.dtype == np.int64


def test_empty_batch_is_a_programming_error_not_a_silent_empty_array():
    with pytest.raises(ValueError):
        tok.encode([])
