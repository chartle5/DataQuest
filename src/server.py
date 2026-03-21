"""Flask server — serves both the API and the frontend static files.

Endpoints:
- POST /api/rank     -> ranked candidates for a condition
- POST /api/train    -> trigger model training
- GET  /api/metrics  -> latest evaluation metrics
- GET  /api/trials   -> list cached trials
- POST /api/verify   -> verify top-N candidates against external API
- GET  /*            -> frontend static files
"""
from pathlib import Path
import json
import logging
import os
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory

from src.patients.features import build_patient_profiles, PatientProfile
from src.trials.api import fetch_trials
from src.trials.rag import build_trial_rag_features
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


# ---------- trial selection ----------

def _trial_richness(trial) -> int:
    """Score a trial by how many structured criteria it defines.

    Trials with more non-null constraints are more useful for matching because
    they let the scoring function differentiate candidates.
    """
    c = trial.criteria
    score = 0
    if c.min_age is not None:
        score += 1
    if c.max_age is not None:
        score += 1
    if c.sex_allowed not in (None, "all"):
        score += 1
    if c.requires_t2d:
        score += 2  # strong signal
    if c.hba1c_min is not None:
        score += 2
    if c.hba1c_max is not None:
        score += 1
    if c.excludes_renal_failure:
        score += 1
    if c.excludes_recent_stroke:
        score += 1
    if c.excludes_insulin_pump:
        score += 1
    if c.entities:
        score += min(len(c.entities), 3)
    return score


def _select_best_trial(trials):
    """Pick the trial with the richest structured criteria for ranking."""
    if not trials:
        return None
    return max(trials, key=_trial_richness)


# ---------- reason / confidence helpers ----------

def _build_reasons(feat: Dict[str, float]) -> List[str]:
    """Human-readable reason codes per patient-trial pair."""
    reasons = []
    if feat.get("age_match") == 1.0:
        reasons.append("Age within required range")
    elif feat.get("age_gap", 0) > 0:
        reasons.append(f"Age outside range (gap {feat['age_gap']:.2f})")

    if feat.get("sex_match") == 1.0:
        reasons.append("Sex matches trial requirement")

    if feat.get("t2d_present") == 1.0:
        reasons.append("Key condition (T2D) present")
    else:
        reasons.append("Key condition (T2D) not found")

    if feat.get("hba1c_above_min") == 1.0 and feat.get("missing_hba1c", 0) == 0:
        reasons.append("HbA1c above minimum threshold")
    if feat.get("hba1c_below_max") == 1.0 and feat.get("missing_hba1c", 0) == 0:
        reasons.append("HbA1c within maximum threshold")
    if feat.get("missing_hba1c") == 1.0:
        reasons.append("HbA1c data unavailable")

    if feat.get("renal_exclusion_hit") == 1.0:
        reasons.append("EXCLUDED: renal failure present")
    if feat.get("stroke_exclusion_hit") == 1.0:
        reasons.append("EXCLUDED: recent stroke history")
    if feat.get("insulin_pump_conflict") == 1.0:
        reasons.append("EXCLUDED: insulin pump conflict")

    if feat.get("lab_completeness", 0) >= 0.5:
        reasons.append("Good lab data completeness")
    elif feat.get("lab_completeness", 0) < 0.2:
        reasons.append("Poor lab data completeness")

    return reasons


def _compute_confidence(score: float, feat: Dict[str, float]) -> float:
    """Compute a numeric confidence score 0-100 from raw score + features.

    Uses sigmoid scaling with adjustments for missing data and exclusions.
    """
    # Sigmoid-scale the raw score into 0-100 range
    # Raw heuristic scores typically fall in [-0.5, 1.0]
    scaled = 1.0 / (1.0 + np.exp(-6.0 * (score - 0.3)))
    confidence = scaled * 100.0

    # Penalize missing data
    missing = feat.get("unknown_field_count", 0)
    confidence -= missing * 8.0

    # Hard exclusions cap confidence
    exclusions = feat.get("exclusion_conflict_count", 0)
    if exclusions > 0:
        confidence = min(confidence, 15.0)

    return round(max(0.0, min(100.0, confidence)), 2)


def _confidence_label(confidence_score: float) -> str:
    """Map numeric confidence to categorical label."""
    if confidence_score >= 70:
        return "high"
    if confidence_score >= 45:
        return "moderate"
    if confidence_score >= 20:
        return "uncertain"
    return "ineligible"


# ---------- scoring ----------

def _heuristic_score(df, feature_cols):
    """Weighted heuristic score with continuous features for differentiation."""
    score = np.zeros(len(df))

    # Primary match signals (binary)
    weights = {
        "age_match": 0.12,
        "sex_match": 0.04,
        "t2d_present": 0.18,
        "hba1c_above_min": 0.08,
        "hba1c_below_max": 0.04,
        "inclusion_satisfied_count": 0.06,  # 0-5 range, so 0.06 * 5 = 0.30 max
        "diagnosis_overlap_score": 0.08,
        "rag_sim_max": 0.04,
        "rag_sim_mean": 0.02,
    }
    # Continuous tie-breaking features (patient-intrinsic)
    tiebreaker_weights = {
        "lab_completeness": 0.06,
        "condition_count": 0.03,
        "medication_count": 0.02,
        "age_normalized": 0.01,  # slight signal — older patients more likely T2D
        "hba1c_value": 0.03,
        "insulin_on_med": 0.02,
    }
    # Penalty features
    penalties = {
        "renal_exclusion_hit": -0.30,
        "stroke_exclusion_hit": -0.25,
        "insulin_pump_conflict": -0.20,
        "exclusion_conflict_count": -0.08,
        "missing_hba1c": -0.04,
        "missing_renal_lab": -0.02,
        "age_gap": -0.10,
        "hba1c_gap": -0.05,
        "recent_hospitalization": -0.02,
    }

    for col, w in weights.items():
        if col in df.columns:
            score += df[col].fillna(0).values * w
    for col, w in tiebreaker_weights.items():
        if col in df.columns:
            score += df[col].fillna(0).values * w
    for col, w in penalties.items():
        if col in df.columns:
            score += df[col].fillna(0).values * w

    # Deterministic micro-jitter for tie-breaking (hash-based, not random)
    if "patient_id" in df.columns:
        jitter = df["patient_id"].apply(
            lambda pid: int(hashlib.sha256(str(pid).encode()).hexdigest()[:8], 16) / (16**8) * 1e-6
        ).values
        score += jitter

    return score


# ---------- routes ----------

@APP.route("/api/rank", methods=["POST"])
def api_rank():
    payload = request.get_json() or {}
    condition = payload.get("condition", "type 2 diabetes")
    top_n = int(payload.get("top_n", 10))

    cache_path = config.OUTPUT_DIR / f"trials_{condition.replace(' ', '_')}.json"
    trials = fetch_trials(condition, limit=20, cache_path=cache_path)
    if not trials:
        return jsonify({"error": "no trials found for condition"}), 400

    # Select the trial with richest structured criteria
    trial = _select_best_trial(trials)
    log.info("Selected trial %s (richness=%d) from %d candidates",
             trial.trial_id, _trial_richness(trial), len(trials))

    profiles = _get_profiles()

    # Compute RAG features for the selected trial
    try:
        rag_features = build_trial_rag_features([trial], config.DATA_DIR)
        trial_rag = rag_features.get(trial.trial_id, {"rag_sim_max": 0.0, "rag_sim_mean": 0.0})
    except Exception:
        trial_rag = {"rag_sim_max": 0.0, "rag_sim_mean": 0.0}

    rows = []
    for pid, patient in profiles.items():
        feat = build_match_features(patient, trial.criteria)
        feat.update(trial_rag)
        rows.append({"patient_id": pid, **feat})

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "patient_id"]

    model = _load_model(condition.replace(" ", "_"))
    if model is not None:
        X = df[feature_cols].values.astype(float)
        try:
            scores = model.predict(X)
        except Exception:
            scores = _heuristic_score(df, feature_cols)
    else:
        scores = _heuristic_score(df, feature_cols)

    df["score"] = scores
    df_sorted = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    candidates = []
    for rank_idx, row in df_sorted.iterrows():
        pid = row["patient_id"]
        feat = {c: float(row[c]) for c in feature_cols}
        names = _patient_names.get(pid, {"first": "", "last": ""})
        raw_score = float(row["score"])
        confidence = _compute_confidence(raw_score, feat)
        candidates.append({
            "patient_id": pid,
            "first_name": names["first"],
            "last_name": names["last"],
            "status": _confidence_label(confidence),
            "confidence_score": confidence,
            "reasons": _build_reasons(feat),
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
    })


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


# ---------- verification endpoint ----------

VERIFY_API_URL = os.environ.get("VERIFY_API_URL", "")
VERIFY_API_KEY = os.environ.get("VERIFY_API_KEY", "")
VERIFY_TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "30"))


def _select_top_n(candidates: List[Dict], n: int) -> List[Dict]:
    """Select the top N candidates (already sorted by rank)."""
    return candidates[:n]


def _build_verification_payload(candidate: Dict) -> Dict[str, Any]:
    """Extract minimum required personal fields for verification."""
    return {
        "patient_id": candidate["patient_id"],
        "first_name": candidate["first_name"],
        "last_name": candidate["last_name"],
    }


def _call_verification_api(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Post candidate details to the verification API and return results.

    Raises on HTTP or connection errors so the caller can return a clean error.
    """
    if not VERIFY_API_URL:
        raise ValueError("VERIFY_API_URL environment variable is not configured")

    headers = {"Content-Type": "application/json"}
    if VERIFY_API_KEY:
        headers["Authorization"] = f"Bearer {VERIFY_API_KEY}"

    resp = http_requests.post(
        VERIFY_API_URL,
        json={"candidates": payloads},
        headers=headers,
        timeout=VERIFY_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


@APP.route("/api/verify", methods=["POST"])
def api_verify():
    """Verify top-N ranked candidates against an external verification API.

    Request body:
        {
            "condition": "type 2 diabetes",
            "top_n": 5,           // how many top candidates to verify
            "candidates": [...]   // optional — pre-ranked candidate list
        }

    If "candidates" is omitted, the endpoint re-runs /api/rank internally.
    """
    payload = request.get_json() or {}
    top_n = payload.get("top_n")
    candidates = payload.get("candidates")
    condition = payload.get("condition", "type 2 diabetes")

    # --- validation ---
    if top_n is None:
        return jsonify({"error": "top_n is required"}), 400
    try:
        top_n = int(top_n)
        if top_n < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "top_n must be a positive integer"}), 400

    # If no pre-ranked candidates provided, generate them
    if not candidates:
        cache_path = config.OUTPUT_DIR / f"trials_{condition.replace(' ', '_')}.json"
        trials = fetch_trials(condition, limit=20, cache_path=cache_path)
        if not trials:
            return jsonify({"error": "no trials found for condition"}), 400

        trial = _select_best_trial(trials)
        profiles = _get_profiles()

        try:
            rag_features = build_trial_rag_features([trial], config.DATA_DIR)
            trial_rag = rag_features.get(trial.trial_id, {"rag_sim_max": 0.0, "rag_sim_mean": 0.0})
        except Exception:
            trial_rag = {"rag_sim_max": 0.0, "rag_sim_mean": 0.0}

        rows = []
        for pid, patient in profiles.items():
            feat = build_match_features(patient, trial.criteria)
            feat.update(trial_rag)
            rows.append({"patient_id": pid, **feat})

        df = pd.DataFrame(rows)
        feature_cols = [c for c in df.columns if c != "patient_id"]

        model = _load_model(condition.replace(" ", "_"))
        if model is not None:
            X = df[feature_cols].values.astype(float)
            try:
                scores = model.predict(X)
            except Exception:
                scores = _heuristic_score(df, feature_cols)
        else:
            scores = _heuristic_score(df, feature_cols)

        df["score"] = scores
        df_sorted = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

        candidates = []
        for _, row in df_sorted.iterrows():
            pid = row["patient_id"]
            feat = {c: float(row[c]) for c in feature_cols}
            names = _patient_names.get(pid, {"first": "", "last": ""})
            raw_score = float(row["score"])
            confidence = _compute_confidence(raw_score, feat)
            candidates.append({
                "patient_id": pid,
                "first_name": names["first"],
                "last_name": names["last"],
                "status": _confidence_label(confidence),
                "confidence_score": confidence,
                "reasons": _build_reasons(feat),
            })

    # Select top N
    selected = _select_top_n(candidates, top_n)
    if not selected:
        return jsonify({"error": "no candidates available to verify"}), 400

    log.info("Verification: sending top %d candidates", len(selected))

    # Validate minimum required fields
    for c in selected:
        if not c.get("patient_id") or not c.get("first_name") or not c.get("last_name"):
            return jsonify({"error": "candidate missing required fields (patient_id, first_name, last_name)"}), 400

    # Build minimal payloads
    verification_payloads = [_build_verification_payload(c) for c in selected]

    # Call external verification API
    try:
        api_results = _call_verification_api(verification_payloads)
    except ValueError as e:
        log.error("Verification config error: %s", e)
        return jsonify({"error": str(e)}), 500
    except http_requests.Timeout:
        log.error("Verification API timed out")
        return jsonify({"error": "verification API timed out"}), 504
    except http_requests.RequestException as e:
        log.error("Verification API request failed: %s", e)
        return jsonify({"error": f"verification API request failed: {e}"}), 502

    log.info("Verification: received %d results", len(api_results))

    # Map results back to candidates
    result_map = {r.get("patient_id"): r for r in api_results} if api_results else {}
    verified = []
    for rank_idx, cand in enumerate(selected, start=1):
        pid = cand["patient_id"]
        verification = result_map.get(pid, {"status": "no_response"})
        verified.append({
            "patient_id": pid,
            "first_name": cand["first_name"],
            "last_name": cand["last_name"],
            "rank": rank_idx,
            "confidence_score": cand["confidence_score"],
            "verification_result": verification,
        })

    return jsonify({"verified_candidates": verified})


# ---------- frontend ----------

@APP.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@APP.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(FRONTEND_DIR), path)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8080, debug=True)
