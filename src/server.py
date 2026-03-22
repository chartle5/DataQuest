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
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory

from src.patients.features import build_patient_profiles, PatientProfile
from src.trials.api import fetch_trials
from src.trials.rag import build_trial_rag_features
from src.matching.features import build_match_features, _detect_mode
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


def _clean_name(name: str) -> str:
    """Strip trailing digits from Synthea-generated names (e.g. 'Cordell41' -> 'Cordell')."""
    return re.sub(r'\d+$', '', name).strip()


def _get_profiles() -> Dict[str, PatientProfile]:
    global _profiles_cache, _patient_names
    if _profiles_cache:
        return _profiles_cache

    patients_path = str(_resolve_data_path("patients.csv"))
    patients_df = pd.read_csv(patients_path)
    for _, row in patients_df.iterrows():
        pid = row["Id"]
        _patient_names[pid] = {
            "first": _clean_name(str(row.get("FIRST", ""))),
            "last": _clean_name(str(row.get("LAST", ""))),
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

def _trial_richness(trial, mode: str = "diabetes") -> int:
    """Score a trial by how many structured criteria it defines.

    Trials with more non-null constraints are more useful for matching because
    they let the scoring function differentiate candidates.
    """
    c = trial.criteria
    score = 0
    if c.min_age is not None:
        score += 2 if mode == "cancer" else 1
    if c.max_age is not None:
        score += 2 if mode == "cancer" else 1
    if c.sex_allowed not in (None, "all"):
        score += 1
    if mode == "diabetes":
        if c.requires_t2d:
            score += 2  # strong signal for diabetes
        if c.hba1c_min is not None:
            score += 2
        if c.hba1c_max is not None:
            score += 1
    if mode == "cancer":
        # Cancer trials: value entity and condition richness
        if c.entities:
            score += min(len(c.entities), 5)
        # Conditions list contains cancer keywords
        cond_text = " ".join(trial.conditions).lower()
        if any(kw in cond_text for kw in ("cancer", "carcinoma", "neoplasm", "lymphoma", "melanoma", "leukemia")):
            score += 3
    if c.excludes_renal_failure:
        score += 1
    if c.excludes_recent_stroke:
        score += 1
    if c.excludes_insulin_pump:
        score += 1
    if c.entities and mode == "diabetes":
        score += min(len(c.entities), 3)
    return score


# Pinned trial IDs per mode — ensures consistent trial selection across requests.
# When a pinned trial exists in the cached list, it is always selected.
PINNED_TRIALS = {
    "cancer": "NCT06767046",
}


def _select_best_trial(trials, mode: str = "diabetes", profiles=None):
    """Pick the trial with the richest structured criteria for ranking.

    If a pinned trial ID is configured for the mode and present in the
    trial list, always select it for consistency.

    When patient profiles are available, penalize trials whose age range
    has zero overlap with the actual patient population (avoids selecting
    a trial where every patient is ineligible).
    """
    if not trials:
        return None

    # Check for pinned trial first
    pinned_id = PINNED_TRIALS.get(mode)
    if pinned_id:
        for trial in trials:
            if trial.trial_id == pinned_id:
                return trial

    if profiles:
        ages = [p.age for p in profiles.values()]
        min_patient_age, max_patient_age = min(ages), max(ages)

        def _score(trial):
            base = _trial_richness(trial, mode)
            c = trial.criteria
            trial_min = c.min_age or 0
            trial_max = c.max_age or 120
            # Check if any patient could be in range
            if trial_max < min_patient_age or trial_min > max_patient_age:
                base -= 20  # heavy penalty — no patients can match
            return base

        return max(trials, key=_score)

    return max(trials, key=lambda t: _trial_richness(t, mode))


# ---------- reason / confidence helpers ----------

def _build_reasons(feat: Dict[str, float], mode: str = "diabetes") -> List[str]:
    """Human-readable reason codes per patient-trial pair."""
    reasons = []
    if feat.get("age_match") == 1.0:
        reasons.append("Age within required range")
    elif feat.get("age_gap", 0) > 0:
        gap = feat["age_gap"]
        if gap <= 0.15:
            reasons.append(f"Age slightly outside range (gap {gap:.2f})")
        elif gap <= 0.50:
            reasons.append(f"Age moderately outside range (gap {gap:.2f})")
        else:
            reasons.append(f"Age significantly outside range (gap {gap:.2f})")

    if feat.get("sex_match") == 1.0:
        reasons.append("Sex matches trial requirement")

    if mode == "cancer":
        if feat.get("key_condition_present") == 1.0:
            reasons.append("Cancer-related condition present in patient record")
        else:
            reasons.append("No cancer-related condition found in patient record")
        if feat.get("has_prior_chemo", 0) == 1.0:
            reasons.append("Patient has prior chemotherapy history")
        if feat.get("has_prior_radiation", 0) == 1.0:
            reasons.append("Patient has prior radiation treatment")
        if feat.get("has_metastatic_disease", 0) == 1.0:
            reasons.append("Metastatic disease present")
        tumor_ct = feat.get("tumor_condition_count", 0)
        if tumor_ct > 0:
            reasons.append(f"{int(tumor_ct * 10)} cancer-related condition(s) found")
        chemo_ct = feat.get("cancer_medication_count", 0)
        if chemo_ct > 0:
            reasons.append(f"{int(chemo_ct * 10)} chemo/targeted-therapy medication(s) found")
        if feat.get("has_cancer_exclusion_condition", 0) == 1.0:
            reasons.append("EXCLUDED: comorbidity incompatible with immunotherapy (e.g. transplant, autoimmune)")
        comorbidity = feat.get("cancer_comorbidity_burden", 0)
        if comorbidity >= 0.4:
            reasons.append(f"Significant comorbidity burden ({comorbidity:.1f})")
        elif comorbidity > 0:
            reasons.append(f"Mild comorbidity burden ({comorbidity:.1f})")
    else:
        # Diabetes mode
        if feat.get("key_condition_present") == 1.0:
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


# Track score distribution per ranking call to normalize confidence
_score_stats: Dict[str, float] = {}


def _update_score_stats(scores: np.ndarray) -> None:
    """Cache score distribution stats for confidence normalization."""
    _score_stats["min"] = float(np.min(scores))
    _score_stats["max"] = float(np.max(scores))
    _score_stats["mean"] = float(np.mean(scores))
    _score_stats["std"] = float(np.std(scores)) if len(scores) > 1 else 1.0


def _compute_confidence(score: float, feat: Dict[str, float]) -> float:
    """Compute a numeric confidence score 0-100 from raw score + features.

    Uses percentile-based scaling relative to the current score distribution,
    with adjustments for missing data, exclusions, and key conditions.
    Hard constraints: age/sex mismatch → 0% (ineligible).
    """
    # Hard fail: age outside range → ineligible
    if feat.get("age_match", 1.0) == 0.0:
        return 0.0

    # Hard fail: sex mismatch → ineligible
    if feat.get("sex_match", 1.0) == 0.0:
        return 0.0

    # Normalize score to [0, 1] using distribution stats, then scale to confidence
    s_min = _score_stats.get("min", -1.0)
    s_max = _score_stats.get("max", 1.0)
    s_range = s_max - s_min
    if s_range < 1e-9:
        normalized = 0.5
    else:
        normalized = (score - s_min) / s_range  # 0.0 to 1.0

    # Apply a mild sigmoid to create differentiation (centered at 0.5)
    # This prevents the top candidates from all collapsing to 100%
    scaled = 1.0 / (1.0 + np.exp(-5.0 * (normalized - 0.5)))

    # Map to confidence range: top candidate ~85-92%, not 100%
    confidence = 20.0 + scaled * 72.0  # range: 20% to 92%

    # --- Feature bonuses for differentiation among eligible candidates ---
    if feat.get("key_condition_present", 0.0) == 1.0:
        # Cancer-specific bonuses
        if feat.get("has_metastatic_disease", 0.0) == 1.0:
            confidence += 3.0
        if feat.get("has_prior_radiation", 0.0) == 1.0:
            confidence += 1.5
        tumor_ct = feat.get("tumor_condition_count", 0.0)
        confidence += tumor_ct * 4.0
        chemo_ct = feat.get("cancer_medication_count", 0.0)
        confidence += chemo_ct * 3.0

        # Diabetes-specific bonuses
        hba1c_val = feat.get("hba1c_value", 0.0)  # scaled 0-1
        if hba1c_val > 0:
            confidence += hba1c_val * 3.0  # higher HbA1c = stronger T2D signal
        if feat.get("hba1c_above_min", 0.0) == 1.0:
            confidence += 1.5
        if feat.get("insulin_on_med", 0.0) == 1.0:
            confidence += 1.0

        # Shared continuous bonuses
        diag_overlap = feat.get("diagnosis_overlap_score", 0.0)
        confidence += diag_overlap * 2.5
        lab = feat.get("lab_completeness", 0.0)
        confidence += lab * 2.0
        cond_ct = feat.get("condition_count", 0.0)
        confidence += cond_ct * 1.5
        med_ct = feat.get("medication_count", 0.0)
        confidence += med_ct * 1.5
        age_norm = feat.get("age_normalized", 0.0)
        confidence += age_norm * 1.0

    # Penalize missing data
    missing = feat.get("unknown_field_count", 0)
    confidence -= missing * 10.0

    # Hard exclusions cap confidence
    exclusions = feat.get("exclusion_conflict_count", 0)
    if exclusions > 0:
        confidence = min(confidence, 15.0)

    # Missing key condition → soft cap
    if feat.get("key_condition_present", 0.0) == 0.0:
        confidence = min(confidence, 15.0)

    # Cancer-specific exclusion condition → cap
    if feat.get("has_cancer_exclusion_condition", 0.0) == 1.0:
        confidence = min(confidence, 15.0)

    # Cancer comorbidity burden penalty
    comorbidity = feat.get("cancer_comorbidity_burden", 0.0)
    if comorbidity > 0:
        confidence -= comorbidity * 20.0

    # Additional penalty for missing HbA1c in diabetes context
    if feat.get("missing_hba1c", 0.0) == 1.0:
        confidence -= 8.0

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

def _heuristic_score(df, feature_cols, mode: str = "diabetes"):
    """Weighted heuristic score with continuous features for differentiation."""
    score = np.zeros(len(df))

    if mode == "cancer":
        weights = {
            "age_match": 0.20,
            "sex_match": 0.06,
            "key_condition_present": 0.25,
            "has_metastatic_disease": 0.10,
            "has_prior_chemo": 0.08,
            "cancer_medication_count": 0.06,
            "tumor_condition_count": 0.05,
            "diagnosis_overlap_score": 0.10,
            "inclusion_satisfied_count": 0.04,
            "rag_sim_max": 0.04,
            "rag_sim_mean": 0.02,
        }
        tiebreaker_weights = {
            "lab_completeness": 0.04,
            "condition_count": 0.03,
            "medication_count": 0.02,
            "age_normalized": 0.01,
        }
        penalties = {
            "renal_exclusion_hit": -0.20,
            "stroke_exclusion_hit": -0.15,
            "has_cancer_exclusion_condition": -0.30,
            "cancer_comorbidity_burden": -0.15,
            "exclusion_conflict_count": -0.06,
            "age_gap": -0.30,
            "missing_renal_lab": -0.02,
            "recent_hospitalization": -0.01,
        }
    else:
        # Diabetes mode (default)
        weights = {
            "age_match": 0.25,
            "sex_match": 0.04,
            "key_condition_present": 0.20,
            "hba1c_above_min": 0.12,
            "hba1c_below_max": 0.04,
            "inclusion_satisfied_count": 0.06,
            "diagnosis_overlap_score": 0.08,
            "rag_sim_max": 0.04,
            "rag_sim_mean": 0.02,
        }
        tiebreaker_weights = {
            "lab_completeness": 0.06,
            "condition_count": 0.03,
            "medication_count": 0.02,
            "age_normalized": 0.01,
            "hba1c_value": 0.03,
            "insulin_on_med": 0.02,
        }
        penalties = {
            "renal_exclusion_hit": -0.35,
            "stroke_exclusion_hit": -0.25,
            "insulin_pump_conflict": -0.20,
            "exclusion_conflict_count": -0.08,
            "missing_hba1c": -0.04,
            "missing_renal_lab": -0.02,
            "age_gap": -0.35,
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

    # Hard floor: age or sex outside range → minimum score
    if "age_match" in df.columns:
        age_fail = df["age_match"].fillna(1).values == 0.0
        score[age_fail] = -1.0
    if "sex_match" in df.columns:
        sex_fail = df["sex_match"].fillna(1).values == 0.0
        score[sex_fail] = -1.0

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
    mode = _detect_mode(condition)

    cache_path = config.OUTPUT_DIR / f"trials_{condition.replace(' ', '_')}.json"
    trials = fetch_trials(condition, limit=20, cache_path=cache_path, mode=mode)
    if not trials:
        return jsonify({"error": "no trials found for condition"}), 400

    profiles = _get_profiles()

    # Select the trial with richest structured criteria
    trial = _select_best_trial(trials, mode=mode, profiles=profiles)
    log.info("Selected trial %s (richness=%d, mode=%s) from %d candidates",
             trial.trial_id, _trial_richness(trial, mode), mode, len(trials))

    # Derive condition keywords from the trial's conditions list
    condition_keywords = [c.lower() for c in trial.conditions] if trial.conditions else None

    # Compute RAG features for the selected trial
    try:
        rag_features = build_trial_rag_features([trial], config.DATA_DIR)
        trial_rag = rag_features.get(trial.trial_id, {"rag_sim_max": 0.0, "rag_sim_mean": 0.0})
    except Exception:
        trial_rag = {"rag_sim_max": 0.0, "rag_sim_mean": 0.0}

    rows = []
    for pid, patient in profiles.items():
        feat = build_match_features(patient, trial.criteria,
                                    condition_keywords=condition_keywords, mode=mode)
        feat.update(trial_rag)
        rows.append({"patient_id": pid, **feat})

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "patient_id"]

    model = _load_model(condition.replace(" ", "_"))
    if model is not None:
        # Align features to the model's expected columns to avoid misalignment
        _mf = getattr(model, 'feature_names_in_', None)
        if _mf is None or len(_mf) == 0:
            _mf = getattr(model, 'feature_name_', None)
        model_features = list(_mf) if _mf is not None and len(_mf) > 0 else []
        if model_features and all(f in df.columns for f in model_features):
            X = df[model_features].values.astype(float)
            try:
                scores = model.predict(X)
            except Exception:
                scores = _heuristic_score(df, feature_cols, mode=mode)
        else:
            log.warning("Model features not aligned with data columns, using heuristic")
            scores = _heuristic_score(df, feature_cols, mode=mode)
    else:
        scores = _heuristic_score(df, feature_cols, mode=mode)

    df["score"] = scores
    _update_score_stats(scores)
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
            "reasons": _build_reasons(feat, mode=mode),
        })

    # Sort candidates by confidence score descending
    candidates.sort(key=lambda c: c["confidence_score"], reverse=True)

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
    mode = request.args.get("mode", "")
    # Try mode-specific metrics first
    if mode:
        mode_path = config.OUTPUT_DIR / f"evaluation_metrics_{mode}.json"
        if mode_path.exists():
            with open(mode_path) as f:
                return jsonify(json.load(f))
    # Fallback to generic metrics
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
            _mf = getattr(model, 'feature_names_in_', None)
            if _mf is None or len(_mf) == 0:
                _mf = getattr(model, 'feature_name_', None)
            model_features = list(_mf) if _mf is not None and len(_mf) > 0 else []
            if model_features and all(f in df.columns for f in model_features):
                X = df[model_features].values.astype(float)
                try:
                    scores = model.predict(X)
                except Exception:
                    scores = _heuristic_score(df, feature_cols)
            else:
                scores = _heuristic_score(df, feature_cols)
        else:
            scores = _heuristic_score(df, feature_cols)

        df["score"] = scores
        _update_score_stats(scores)
        df_sorted = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

        mode = _detect_mode(condition)
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
                "reasons": _build_reasons(feat, mode=mode),
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
