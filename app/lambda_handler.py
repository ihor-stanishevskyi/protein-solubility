"""Adapter so the same FastAPI app answers Lambda's event shape."""
from mangum import Mangum

from .main import app

handler = Mangum(app, lifespan="off")
