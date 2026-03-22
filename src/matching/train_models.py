"""Training utilities to build patient-trial pairs and train a TrialRanker per condition."""
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from src import config
from src.patients.features import build_patient_profiles
from src.trials.api import fetch_trials
from src.matching.features import build_match_features, _detect_mode
from src.matching.ranker import TrialRanker

# Features used to define _rule_label() — must be excluded from training X
LABEL_FEATURES = {
    "key_condition_present", "inclusion_satisfied_count",
    "renal_exclusion_hit", "stroke_exclusion_hit", "insulin_pump_conflict",
    "missing_hba1c", "missing_renal_lab",
    "age_match", "sex_match",
    "has_cancer_exclusion_condition",
    "has_prior_chemo", "has_metastatic_disease",
}


def _rule_label(features: dict, mode: str = "diabetes") -> int:
    """Graded relevance label for lambdarank: 0=excluded, 1=poor, 2=partial, 3=good.

    Higher values indicate better patient-trial match quality.
    Mode-aware: uses HbA1c for diabetes, cancer-specific features for cancer.
    """
    # Hard demographic exclusions -> grade 0
    if features.get("age_match", 1.0) == 0.0:
        return 0
    if features.get("sex_match", 1.0) == 0.0:
        return 0

    # Hard clinical exclusions -> grade 0
    if features.get("renal_exclusion_hit", 0.0) == 1.0:
        return 0
    if features.get("stroke_exclusion_hit", 0.0) == 1.0:
        return 0
    if features.get("insulin_pump_conflict", 0.0) == 1.0:
        return 0

    has_key_condition = features.get("key_condition_present", 0.0) == 1.0

    # No key condition -> grade 0
    if not has_key_condition:
        return 0

    lab_complete = features.get("lab_completeness", 0.0) >= 0.5
    good_overlap = features.get("diagnosis_overlap_score", 0.0) >= 0.3

    if mode == "cancer":
        has_chemo = features.get("has_prior_chemo", 0.0) == 1.0
        has_metastatic = features.get("has_metastatic_disease", 0.0) == 1.0
        tumor_count = features.get("tumor_condition_count", 0.0)
        has_exclusion = features.get("has_cancer_exclusion_condition", 0.0) == 1.0
        comorbidity = features.get("cancer_comorbidity_burden", 0.0)

        # Hard cancer exclusion (transplant, autoimmune, etc.) -> grade 0
        if has_exclusion:
            return 0

        # Grade 3: chemo + (metastatic or good overlap) + lab complete + low comorbidity
        if has_chemo and (has_metastatic or good_overlap) and lab_complete and comorbidity < 0.4:
            return 3
        # Grade 2: at least one strong signal + low-moderate comorbidity
        if (has_chemo or has_metastatic or (good_overlap and tumor_count >= 0.2)) and comorbidity < 0.6:
            return 2
        # Grade 1: key condition present but insufficient evidence or comorbidity concern
        return 1

    # --- Diabetes mode ---
    has_missing = (features.get("missing_hba1c", 0.0) + features.get("missing_renal_lab", 0.0)) > 0
    hba1c_ok = (features.get("hba1c_above_min", 0.0) == 1.0 or
                features.get("hba1c_below_max", 0.0) == 1.0)

    # Key condition but missing critical lab data and no HbA1c -> grade 1
    if has_missing and not hba1c_ok:
        return 1

    # Grade 3: all criteria met
    if hba1c_ok and not has_missing and lab_complete and good_overlap:
        return 3

    # Grade 2: key condition + HbA1c ok but not all supporting data
    if hba1c_ok and (not has_missing or lab_complete):
        return 2

    # Grade 1: key condition but incomplete data
    return 1


def train_for_condition(condition_query: str, trials_limit: int = 50, out_dir: Path = None):
    out_dir = out_dir or (config.OUTPUT_DIR / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = _detect_mode(condition_query)

    profiles = build_patient_profiles(
        patients_path=str(config.DATA_DIR / "patients.csv"),
        conditions_path=str(config.DATA_DIR / "conditions.csv"),
        medications_path=str(config.DATA_DIR / "medications.csv"),
        observations_path=str(config.DATA_DIR / "observations.csv"),
        encounters_path=str(config.DATA_DIR / "encounters.csv"),
    )

    cache_path = config.OUTPUT_DIR / f"trials_{condition_query.replace(' ', '_')}.json"
    trials = fetch_trials(condition_query, limit=trials_limit, cache_path=cache_path, mode=mode)

    rows = []
    for trial in trials:
        condition_keywords = [c.lower() for c in trial.conditions] if trial.conditions else None
        for patient in profiles.values():
            feat = build_match_features(patient, trial.criteria,
                                        condition_keywords=condition_keywords, mode=mode)
            label = _rule_label(feat, mode=mode)
            rows.append({"trial_id": trial.trial_id, "patient_id": patient.patient_id, **feat, "label": label})

    df = pd.DataFrame(rows)
    # Exclude label-defining features to prevent leakage
    feature_cols = [c for c in df.columns
                    if c not in ("trial_id", "patient_id", "label")
                    and c not in LABEL_FEATURES]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(float)

    # --- train / test split by trial ---
    trial_ids = df["trial_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(trial_ids)
    split_idx = max(1, int(len(trial_ids) * 0.8))
    train_trials = set(trial_ids[:split_idx])

    train_mask = df["trial_id"].isin(train_trials).values
    X_train, y_train = X[train_mask], y[train_mask]
    group = df[df["trial_id"].isin(train_trials)].groupby("trial_id").size().tolist()

    ranker = TrialRanker()
    ranker.fit(X_train, y_train, group, feature_names=feature_cols)

    # persist model
    try:
        import joblib

        model_path = out_dir / f"{condition_query.replace(' ', '_')}_model.pkl"
        joblib.dump(ranker.model, model_path)
    except Exception:
        model_path = None

    # --- evaluate on test set ---
    test_mask = ~train_mask
    if test_mask.any():
        y_test = y[test_mask]
        scores_test = ranker.predict(X[test_mask])
        test_trial_ids = df.loc[test_mask, "trial_id"].values
        from src.pipelines.build_pairs import _compute_metrics
        metrics = _compute_metrics(y_test, scores_test, trial_ids=test_trial_ids)
        import json
        metrics_path = config.OUTPUT_DIR / f"evaluation_metrics_{mode}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return model_path


if __name__ == "__main__":
    m = train_for_condition("type 2 diabetes", trials_limit=20)
    print("Saved model:", m)
