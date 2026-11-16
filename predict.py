"""Load the trained artifact and run title-type inference."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from model import FEATURE_COLUMNS


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/netflix_type.joblib"))


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_title(features):
    """Return one class prediction from a feature dictionary."""
    row = {column: features.get(column) for column in FEATURE_COLUMNS}
    for column in ("description", "listed_in", "country"):
        if row[column] is None:
            row[column] = ""
    if row["rating"] is None:
        row["rating"] = "Unknown"

    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    frame["release_year"] = pd.to_numeric(frame["release_year"], errors="coerce")
    return str(load_model().predict(frame)[0])
