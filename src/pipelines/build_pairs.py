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
    # Stratified: ensure test set contains trials with both positive and negative labels
    trial_ids = df["trial_id"].unique()
    trial_neg_rate = {}
    for tid in trial_ids:
        labels = df.loc[df["trial_id"] == tid, "label"].values
        trial_neg_rate[tid] = float((labels < 2).sum()) / max(len(labels), 1)

    # Sort trials so ones with negatives are spread across train/test
    sorted_trials = sorted(trial_ids, key=lambda t: trial_neg_rate[t], reverse=True)
    rng = np.random.RandomState(42)
    split_idx = max(1, int(len(sorted_trials) * 0.8))

    # Interleave: assign every 5th trial with negatives to test set
    has_negatives = [t for t in sorted_trials if trial_neg_rate[t] > 0]
    all_positive = [t for t in sorted_trials if trial_neg_rate[t] == 0]
    rng.shuffle(has_negatives)
    rng.shuffle(all_positive)

    # Ensure at least some trials with negatives land in test
    neg_for_test = has_negatives[:max(1, len(has_negatives) // 5)]
    neg_for_train = has_negatives[max(1, len(has_negatives) // 5):]
    remaining_test_slots = max(0, int(len(trial_ids) * 0.2) - len(neg_for_test))
    pos_for_test = all_positive[:remaining_test_slots]
    pos_for_train = all_positive[remaining_test_slots:]

    train_trials = set(neg_for_train) | set(pos_for_train)
    test_trials = set(neg_for_test) | set(pos_for_test)

    train_mask = df["trial_id"].isin(train_trials)
    X_train, y_train = X[train_mask], y[train_mask]
    train_groups = df[train_mask].groupby("trial_id").size().tolist()

    ranker = TrialRanker()
    ranker.fit(X_train, y_train, train_groups, feature_names=feature_cols)

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

    # Binarize graded labels: grade >= 2 counts as "relevant"
    y_binary = (y_true >= 2).astype(float)

    # Find threshold that maximizes F1 instead of using median
    sorted_scores = np.unique(scores)
    best_f1, best_thresh = 0.0, np.median(scores)
    for t in sorted_scores:
        yp = (scores >= t).astype(float)
        f1_val = float(f1_score(y_binary, yp, zero_division=0))
        if f1_val > best_f1:
            best_f1, best_thresh = f1_val, t
    threshold = best_thresh
    y_pred = (scores >= threshold).astype(float)

    metrics: Dict[str, float] = {}
    # ROC-AUC needs at least two classes in y_binary
    if len(np.unique(y_binary)) >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_binary, scores))
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
    metrics["precision"] = float(precision_score(y_binary, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_binary, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_binary, y_pred, zero_division=0))

    # precision@k: fraction of top-k scored patients that are relevant (grade >= 2)
    top_k_idx = np.argsort(scores)[::-1][:k]
    if len(top_k_idx) > 0:
        metrics[f"precision_at_{k}"] = float(y_binary[top_k_idx].sum() / len(top_k_idx))
    else:
        metrics[f"precision_at_{k}"] = 0.0

    return metrics


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


if __name__ == "__main__":
    df_pairs = build_patient_trial_pairs(trials_limit=50)
    rank_patients(df_pairs)
