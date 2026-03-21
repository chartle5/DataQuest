import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ..config import DATA_DIR, OUTPUT_DIR, PAIRS_OUTPUT_PATH, RANKED_OUTPUT_PATH, TRIALS_CACHE_PATH
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
    PAIRS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PAIRS_OUTPUT_PATH, index=False)
    return df


def rank_patients(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("trial_id", "patient_id", "label")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(float)

    group_sizes = df.groupby("trial_id").size().tolist()

    # --- train / test split by trial groups ---
    trial_ids = df["trial_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(trial_ids)
    split_idx = max(1, int(len(trial_ids) * 0.8))
    train_trials = set(trial_ids[:split_idx])
    test_trials = set(trial_ids[split_idx:])

    train_mask = df["trial_id"].isin(train_trials)
    X_train, y_train = X[train_mask], y[train_mask]
    train_groups = df[train_mask].groupby("trial_id").size().tolist()

    ranker = TrialRanker()
    ranker.fit(X_train, y_train, train_groups)

    # score all rows (train + test) for output
    df["score"] = ranker.predict(X)

    # --- evaluation on test set ---
    if test_trials:
        test_mask = df["trial_id"].isin(test_trials)
        y_test = y[test_mask]
        scores_test = df.loc[test_mask, "score"].values
        metrics = _compute_metrics(y_test, scores_test)
        metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
        import json
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    ranked = df.sort_values(["trial_id", "score"], ascending=[True, False])
    RANKED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(RANKED_OUTPUT_PATH, index=False)
    return ranked


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Compute evaluation metrics for the ranking model."""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    # binarize scores for classification metrics
    threshold = np.median(scores)
    y_pred = (scores >= threshold).astype(float)

    metrics: Dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = 0.0
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # precision@k
    top_k_idx = np.argsort(scores)[::-1][:k]
    if len(top_k_idx) > 0:
        metrics[f"precision_at_{k}"] = float(y_true[top_k_idx].sum() / len(top_k_idx))
    else:
        metrics[f"precision_at_{k}"] = 0.0

    return metrics


def _rule_label(features: dict) -> int:
    """3-class labeling: 0=ineligible, 1=eligible, 2=uncertain."""
    # Hard exclusions -> 0
    if features.get("renal_exclusion_hit", 0.0) == 1.0:
        return 0
    if features.get("stroke_exclusion_hit", 0.0) == 1.0:
        return 0
    if features.get("insulin_pump_conflict", 0.0) == 1.0:
        return 0

    # Missing critical data -> uncertain
    if features.get("missing_hba1c", 0.0) == 1.0 or features.get("missing_renal_lab", 0.0) == 1.0:
        return 2

    # All inclusion criteria met -> eligible
    if (features.get("age_match", 0.0) == 1.0
            and features.get("sex_match", 1.0) == 1.0
            and features.get("t2d_present", 0.0) == 1.0):
        return 1

    return 0


if __name__ == "__main__":
    df_pairs = build_patient_trial_pairs(trials_limit=50)
    rank_patients(df_pairs)
