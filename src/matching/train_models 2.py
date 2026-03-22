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
            label = 1 if feat.get("age_match", 0.0) == 1.0 and feat.get("sex_match", 1.0) == 1.0 and feat.get("t2d_present", 1.0) == 1.0 else 0
            rows.append({"trial_id": trial.trial_id, "patient_id": patient.patient_id, **feat, "label": label})

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("trial_id", "patient_id", "label")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(float)
    group = df.groupby("trial_id").size().tolist()

    ranker = TrialRanker()
    ranker.fit(X, y, group)

    # persist model
    try:
        import joblib

        model_path = out_dir / f"{condition_query.replace(' ', '_')}_model.pkl"
        joblib.dump(ranker.model, model_path)
    except Exception:
        model_path = None

    return model_path


if __name__ == "__main__":
    m = train_for_condition("type 2 diabetes", trials_limit=20)
    print("Saved model:", m)
