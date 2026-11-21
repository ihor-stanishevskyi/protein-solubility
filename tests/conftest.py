"""Point the service at the stand-in model before the app imports anything.

Every test runs the real path - tokenize, session, pool, softmax - against a
model whose answers are meaningless. A test that needed the trained weights
would be testing the model; these test the service.
"""
import os
import pathlib

STAND_IN = pathlib.Path(__file__).resolve().parent.parent / "models" / "stand-in.onnx"
os.environ["MODEL_PATH"] = str(STAND_IN)
