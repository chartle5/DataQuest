import pandas as pd
import numpy as np
from typing import List

from ..config import DATA_DIR, PAIRS_OUTPUT_PATH, RANKED_OUTPUT_PATH, TRIALS_CACHE_PATH
from ..patients.features import build_patient_profiles
from ..trials.api import fetch_trials
from ..trials.rag import build_trial_rag_features
from ..matching.features import build_match_features
from ..matching.ranker import TrialRanker


def build_patient_trial_pairs(trials_limit: int = 50) -> pd.DataFrame:
    profiles = build_patient_profiles(
        patients_path=str(DATA_DIR / "patients.csv"),
        conditions_path=str(DATA_DIR / "conditions.csv"),
        medications_path=str(DATA_DIR / "medications.csv"),
        observations_path=str(DATA_DIR / "observations.csv"),
        encounters_path=str(DATA_DIR / "encounters.csv"),
    )

    trials = fetch_trials(
        "type 2 diabetes",
        limit=trials_limit,
        cache_path=TRIALS_CACHE_PATH,
    )
    trial_rag = build_trial_rag_features(trials, DATA_DIR)
    rows = []

    for trial in trials:
        for patient in profiles.values():
            features = build_match_features(patient, trial.criteria)
            rag_features = trial_rag.get(trial.trial_id, {"rag_sim_max": 0.0, "rag_sim_mean": 0.0})
            label = _rule_label(features)
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "patient_id": patient.patient_id,
                    **features,
                    **rag_features,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(PAIRS_OUTPUT_PATH, index=False)
    return df


def rank_patients(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("trial_id", "patient_id", "label")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(float)

    group_sizes = df.groupby("trial_id").size().tolist()

    ranker = TrialRanker()
    ranker.fit(X, y, group_sizes)
    df["score"] = ranker.predict(X)

    ranked = df.sort_values(["trial_id", "score"], ascending=[True, False])
    ranked.to_csv(RANKED_OUTPUT_PATH, index=False)
    return ranked


def _rule_label(features: dict) -> int:
    if features["age_match"] == 0.0:
        return 0
    if features["sex_match"] == 0.0:
        return 0
    if features["renal_exclusion_hit"] == 1.0:
        return 0
    if features["stroke_exclusion_hit"] == 1.0:
        return 0
    if features["t2d_present"] == 0.0:
        return 0
    return 1


if __name__ == "__main__":
    df_pairs = build_patient_trial_pairs(trials_limit=50)
    rank_patients(df_pairs)
