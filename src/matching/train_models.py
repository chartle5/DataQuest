"""Training utilities to build patient-trial pairs and train a TrialRanker per condition."""
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from src import config
from src.patients.features import build_patient_profiles
from src.trials.api import fetch_trials
from src.matching.features import build_match_features
from src.matching.ranker import TrialRanker


def _rule_label(features: dict) -> int:
    """Graded relevance label for lambdarank: 0=excluded, 1=poor, 2=partial, 3=good.

    Higher values indicate better patient-trial match quality.
    """
    # Hard exclusions -> grade 0
    if features.get("renal_exclusion_hit", 0.0) == 1.0:
        return 0
    if features.get("stroke_exclusion_hit", 0.0) == 1.0:
        return 0
    if features.get("insulin_pump_conflict", 0.0) == 1.0:
        return 0

    # Count how many inclusion criteria are satisfied
    inc = features.get("inclusion_satisfied_count", 0.0)
    has_missing = (features.get("missing_hba1c", 0.0) + features.get("missing_renal_lab", 0.0)) > 0

    # Good match: key condition present + most inclusions met + no missing data
    if features.get("t2d_present", 0.0) == 1.0 and inc >= 4 and not has_missing:
        return 3
    # Partial match: key condition present or high inclusion count
    if features.get("t2d_present", 0.0) == 1.0 or inc >= 3:
        return 2
    # Poor match: few criteria met, no hard exclusion
    return 1


def train_for_condition(condition_query: str, trials_limit: int = 50, out_dir: Path = None):
    out_dir = out_dir or (config.OUTPUT_DIR / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_patient_profiles(
        patients_path=str(config.DATA_DIR / "patients.csv"),
        conditions_path=str(config.DATA_DIR / "conditions.csv"),
        medications_path=str(config.DATA_DIR / "medications.csv"),
        observations_path=str(config.DATA_DIR / "observations.csv"),
        encounters_path=str(config.DATA_DIR / "encounters.csv"),
    )

    trials = fetch_trials(condition_query, limit=trials_limit, cache_path=(config.OUTPUT_DIR / f"trials_{condition_query.replace(' ', '_')}.json"))

    rows = []
    for trial in trials:
        for patient in profiles.values():
            feat = build_match_features(patient, trial.criteria)
            label = _rule_label(feat)
            rows.append({"trial_id": trial.trial_id, "patient_id": patient.patient_id, **feat, "label": label})

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("trial_id", "patient_id", "label")]
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
        from src.pipelines.build_pairs import _compute_metrics
        metrics = _compute_metrics(y_test, scores_test)
        import json
        metrics_path = config.OUTPUT_DIR / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return model_path


if __name__ == "__main__":
    m = train_for_condition("type 2 diabetes", trials_limit=20)
    print("Saved model:", m)
