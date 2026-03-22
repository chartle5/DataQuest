"""Lightweight Flask server to serve ranking requests.

Endpoints:
- POST /api/rank  -> JSON {condition: 'type2'|'cancer', top_n: int} returns ranked candidates
- POST /api/train -> JSON {condition: 'type2'|'cancer'} triggers model training (if training pipeline available)

If trained model artifacts exist under outputs/models/{condition}_model.pkl they will be used.
Otherwise a heuristic scorer is used.
"""
from pathlib import Path
import json
from typing import Dict, Any

from flask import Flask, request, jsonify

from src.patients.features import build_patient_profiles
from src.trials.api import fetch_trials
from src.matching.features import build_match_features
from src.matching.ranker import TrialRanker
from src import config

APP = Flask(__name__)

MODELS_DIR = config.OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_data_path(filename: str) -> Path:
    # prefer config.DATA_DIR; fallback to repository data/
    p = config.DATA_DIR / filename
    if p.exists():
        return p
    alt = Path("data") / filename
    return alt


def _load_or_train_model(condition: str):
    model_path = MODELS_DIR / f"{condition}_model.pkl"
    if model_path.exists():
        try:
            import joblib

            return joblib.load(model_path)
        except Exception:
            return None
    return None


@APP.route("/api/rank", methods=["POST"])
def api_rank():
    payload = request.get_json() or {}
    condition = payload.get("condition", "type2")
    top_n = int(payload.get("top_n", 10))

    # fetch one representative trial for the condition
    cache_path = config.OUTPUT_DIR / f"trials_{condition}.json"
    trials = fetch_trials(condition, limit=1, cache_path=cache_path)
    if not trials:
        return jsonify({"error": "no trials found for condition"}), 400
    trial = trials[0]

    # build patient profiles
    profiles = build_patient_profiles(
        patients_path=str(_resolve_data_path("patients.csv")),
        conditions_path=str(_resolve_data_path("conditions.csv")),
        medications_path=str(_resolve_data_path("medications.csv")),
        observations_path=str(_resolve_data_path("observations.csv")),
        encounters_path=str(_resolve_data_path("encounters.csv")),
    )

    # build features for each patient
    rows = []
    for pid, patient in profiles.items():
        feat = build_match_features(patient, trial.criteria)
        rows.append({"patient_id": pid, **feat})

    import pandas as pd

    df = pd.DataFrame(rows)

    # Attempt to load trained model
    model = _load_or_train_model(condition)
    if model is not None:
        # model may be a sklearn or lightgbm object with predict
        X = df[[c for c in df.columns if c != "patient_id"]].values.astype(float)
        try:
            scores = model.predict(X)
        except Exception:
            # fallback
            scorer = TrialRanker()
            scores = scorer.predict(X)
    else:
        # heuristic scoring
        df["score"] = (
            df.get("age_match", 0.0) * 0.4
            + df.get("t2d_present", 0.0) * 0.3
            + (1.0 - df.get("missing_hba1c", 0.0)) * 0.3
        )
        scores = df["score"].values

    df["score"] = scores
    df_sorted = df.sort_values("score", ascending=False).head(top_n)

    # Build human-readable rationale from feature values
    out = []
    for _, row in df_sorted.iterrows():
        reasons = []
        if row.get("age_match") == 1.0:
            reasons.append("age matches")
        if row.get("sex_match") == 1.0:
            reasons.append("sex matches")
        if row.get("t2d_present") == 1.0:
            reasons.append("key condition present")
        if row.get("renal_exclusion_hit") == 1.0:
            reasons.append("renal exclusion hit")
        out.append({"patient_id": row["patient_id"], "score": float(row["score"]), "reasons": reasons})

    return jsonify({"trial_id": trial.trial_id, "candidates": out})


@APP.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json() or {}
    condition = payload.get("condition", "type2")
    # delegate to pipeline: for now support type2 and cancer
    from src.matching.train_models import train_for_condition

    model_path = train_for_condition(condition, trials_limit=50)
    if model_path:
        return jsonify({"status": "trained", "model_path": str(model_path)})
    return jsonify({"status": "trained", "model_path": None})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8080, debug=True)
