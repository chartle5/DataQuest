"""Flask server — serves both the API and the frontend static files.

Endpoints:
- POST /api/rank     -> ranked candidates for a condition
- POST /api/train    -> trigger model training
- GET  /api/metrics  -> latest evaluation metrics
- GET  /api/trials   -> list cached trials
- GET  /*            -> frontend static files
"""
from pathlib import Path
import json
import logging
from typing import Any, Dict, List

from flask import Flask, request, jsonify, send_from_directory

from src.patients.features import build_patient_profiles, PatientProfile
from src.trials.api import fetch_trials
from src.matching.features import build_match_features
from src.matching.ranker import TrialRanker
from src import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
APP = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

MODELS_DIR = config.OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- caches ----------
_profiles_cache: Dict[str, PatientProfile] = {}
_patient_names: Dict[str, Dict[str, str]] = {}  # patient_id -> {first, last}


def _get_profiles() -> Dict[str, PatientProfile]:
    global _profiles_cache, _patient_names
    if _profiles_cache:
        return _profiles_cache

    import pandas as pd
    patients_path = str(_resolve_data_path("patients.csv"))
    patients_df = pd.read_csv(patients_path)
    for _, row in patients_df.iterrows():
        pid = row["Id"]
        _patient_names[pid] = {
            "first": str(row.get("FIRST", "")),
            "last": str(row.get("LAST", "")),
        }

    _profiles_cache = build_patient_profiles(
        patients_path=patients_path,
        conditions_path=str(_resolve_data_path("conditions.csv")),
        medications_path=str(_resolve_data_path("medications.csv")),
        observations_path=str(_resolve_data_path("observations.csv")),
        encounters_path=str(_resolve_data_path("encounters.csv")),
    )
    return _profiles_cache


def _resolve_data_path(filename: str) -> Path:
    p = config.DATA_DIR / filename
    if p.exists():
        return p
    alt = Path("data") / filename
    return alt


def _load_model(condition: str):
    model_path = MODELS_DIR / f"{condition}_model.pkl"
    if model_path.exists():
        try:
            import joblib
            return joblib.load(model_path)
        except Exception:
            return None
    return None


def _build_reasons(feat: Dict[str, float]) -> List[str]:
    """Human-readable reason codes per patient-trial pair."""
    reasons = []
    if feat.get("age_match") == 1.0:
        reasons.append("age within required range")
    else:
        gap = feat.get("age_gap", 0)
        reasons.append(f"age outside range (gap={gap:.2f})")
    if feat.get("sex_match") == 1.0:
        reasons.append("sex matches")
    if feat.get("t2d_present") == 1.0:
        reasons.append("key condition present")
    if feat.get("hba1c_above_min") == 1.0:
        reasons.append("HbA1c above minimum threshold")
    if feat.get("missing_hba1c") == 1.0:
        reasons.append("HbA1c data missing (uncertain)")
    if feat.get("renal_exclusion_hit") == 1.0:
        reasons.append("EXCLUDED: renal failure")
    if feat.get("stroke_exclusion_hit") == 1.0:
        reasons.append("EXCLUDED: recent stroke")
    if feat.get("insulin_pump_conflict") == 1.0:
        reasons.append("EXCLUDED: insulin pump conflict")
    return reasons


def _confidence_label(feat: Dict[str, float]) -> str:
    if feat.get("unknown_field_count", 0) >= 1:
        return "uncertain"
    if feat.get("exclusion_conflict_count", 0) >= 1:
        return "ineligible"
    if feat.get("inclusion_satisfied_count", 0) >= 4:
        return "high"
    return "moderate"


# ---------- routes ----------

@APP.route("/api/rank", methods=["POST"])
def api_rank():
    payload = request.get_json() or {}
    condition = payload.get("condition", "type 2 diabetes")
    top_n = int(payload.get("top_n", 10))

    cache_path = config.OUTPUT_DIR / f"trials_{condition.replace(' ', '_')}.json"
    trials = fetch_trials(condition, limit=5, cache_path=cache_path)
    if not trials:
        return jsonify({"error": "no trials found for condition"}), 400
    trial = trials[0]

    profiles = _get_profiles()

    rows = []
    feature_dicts = []
    for pid, patient in profiles.items():
        feat = build_match_features(patient, trial.criteria)
        feature_dicts.append(feat)
        rows.append({"patient_id": pid, **feat})

    import pandas as pd
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "patient_id"]

    model = _load_model(condition.replace(" ", "_"))
    if model is not None:
        import numpy as np
        X = df[feature_cols].values.astype(float)
        try:
            scores = model.predict(X)
        except Exception:
            scores = _heuristic_score(df, feature_cols)
    else:
        scores = _heuristic_score(df, feature_cols)

    df["score"] = scores
    df_sorted = df.sort_values("score", ascending=False).head(top_n)

    candidates = []
    for idx, row in df_sorted.iterrows():
        pid = row["patient_id"]
        feat = {c: float(row[c]) for c in feature_cols}
        names = _patient_names.get(pid, {"first": "", "last": ""})
        candidates.append({
            "patient_id": pid,
            "first": names["first"],
            "last": names["last"],
            "score": round(float(row["score"]), 4),
            "confidence": _confidence_label(feat),
            "reasons": _build_reasons(feat),
            "features": {k: round(v, 3) for k, v in feat.items()},
        })

    # --- write output file ---
    output_path = config.OUTPUT_DIR / f"ranked_{condition.replace(' ', '_')}_top{top_n}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"trial_id": trial.trial_id, "trial_title": trial.title, "candidates": candidates}, f, indent=2)
    log.info("Output written to %s", output_path)

    return jsonify({
        "trial_id": trial.trial_id,
        "trial_title": trial.title,
        "candidates": candidates,
        "output_file": str(output_path),
    })


def _heuristic_score(df, feature_cols):
    import numpy as np
    score = np.zeros(len(df))
    weights = {
        "age_match": 0.15,
        "sex_match": 0.05,
        "t2d_present": 0.20,
        "hba1c_above_min": 0.10,
        "hba1c_below_max": 0.05,
        "inclusion_satisfied_count": 0.10,
        "diagnosis_overlap_score": 0.10,
        "rag_sim_max": 0.05,
    }
    penalties = {
        "renal_exclusion_hit": -0.30,
        "stroke_exclusion_hit": -0.25,
        "insulin_pump_conflict": -0.20,
        "exclusion_conflict_count": -0.10,
        "missing_hba1c": -0.05,
        "age_gap": -0.10,
        "hba1c_gap": -0.05,
    }
    for col, w in weights.items():
        if col in df.columns:
            score += df[col].fillna(0).values * w
    for col, w in penalties.items():
        if col in df.columns:
            score += df[col].fillna(0).values * w
    return score


@APP.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json() or {}
    condition = payload.get("condition", "type 2 diabetes")
    from src.matching.train_models import train_for_condition
    model_path = train_for_condition(condition, trials_limit=50)
    if model_path:
        return jsonify({"status": "trained", "model_path": str(model_path)})
    return jsonify({"status": "trained", "model_path": None})


@APP.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics_path = config.OUTPUT_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "no metrics available — run the training pipeline first"}), 404


@APP.route("/api/trials", methods=["GET"])
def api_trials():
    """Return a list of cached trial files available."""
    trial_files = list(config.OUTPUT_DIR.glob("trials_*.json"))
    result = []
    for tf in trial_files:
        with open(tf) as f:
            data = json.load(f)
        result.append({
            "file": tf.name,
            "count": len(data),
            "trials": [{"trial_id": t["trial_id"], "title": t["title"]} for t in data[:10]],
        })
    return jsonify(result)


# ---------- frontend ----------

@APP.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@APP.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(FRONTEND_DIR), path)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8080, debug=True)
