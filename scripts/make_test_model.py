"""A tiny ONNX model with the real one's signature, for tests and local runs.

It lives in models/ rather than tests/ because the image bakes it in as the
default, and .dockerignore drops tests/ - the first version put it there and
the build would have failed on a missing file.

The service must not need a 140 MB download to be testable. This builds a model
that takes exactly what the real one takes and returns exactly what it returns -
input_ids and attention_mask in, two logits out - so every test exercises the
real code path: tokenize, run, pool, soften into a probability.

Its answers mean nothing. That is the point: a test that depends on the model
being good is testing the model, not the service.
"""
import pathlib

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

VOCAB, DIM = 33, 8
rng = np.random.default_rng(0)

emb = numpy_helper.from_array(rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float32),
                              "esm.embeddings.word_embeddings.weight")
w = numpy_helper.from_array(rng.normal(0, 0.3, (DIM, 2)).astype(np.float32), "head.weight")
b = numpy_helper.from_array(np.zeros(2, dtype=np.float32), "head.bias")

nodes = [
    helper.make_node("Gather", ["esm.embeddings.word_embeddings.weight", "input_ids"],
                     ["h"], axis=0),
    helper.make_node("Cast", ["attention_mask"], ["m_f"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["m_f", "axis2"], ["m3"]),
    helper.make_node("Mul", ["h", "m3"], ["h_masked"]),
    helper.make_node("ReduceSum", ["h_masked", "axis1"], ["summed"], keepdims=0),
    helper.make_node("ReduceSum", ["m3", "axis1"], ["counts"], keepdims=0),
    helper.make_node("Div", ["summed", "counts"], ["pooled"]),
    helper.make_node("MatMul", ["pooled", "head.weight"], ["xw"]),
    helper.make_node("Add", ["xw", "head.bias"], ["logits"]),
]

graph = helper.make_graph(
    nodes, "tiny-solubility-stand-in",
    inputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])],
    outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
    initializer=[emb, w, b,
                 numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
                 numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
out = pathlib.Path(__file__).resolve().parent.parent / "models" / "stand-in.onnx"
save(model, out)
print(f"{out}: {out.stat().st_size/1024:.1f} KB, vocab {VOCAB}")
