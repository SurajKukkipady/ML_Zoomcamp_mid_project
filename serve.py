"""Flask HTTP service for Netflix title-type predictions."""

import math
import os

from flask import Flask, jsonify, request

from model import FEATURE_COLUMNS
from predict import predict_title


app = Flask(__name__)
STRING_FEATURES = ("description", "listed_in", "country", "rating")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    unknown_fields = sorted(set(payload) - set(FEATURE_COLUMNS))
    if unknown_fields:
        return jsonify({"error": f"Unknown fields: {', '.join(unknown_fields)}"}), 400

    for field in STRING_FEATURES:
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            return jsonify({"error": f"{field} must be a string."}), 400

    release_year = payload.get("release_year")
    if release_year is not None:
        try:
            valid_year = (
                not isinstance(release_year, bool)
                and isinstance(release_year, (int, float))
                and math.isfinite(release_year)
            )
        except OverflowError:
            valid_year = False
        if not valid_year:
            return jsonify({"error": "release_year must be a finite number."}), 400

    try:
        predicted_type = predict_title(payload)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid feature value: {error}"}), 400

    return jsonify({"predicted_type": predicted_type}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "9696")))
