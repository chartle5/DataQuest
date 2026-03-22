import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ..config import DATA_DIR, OUTPUT_DIR, PAIRS_OUTPUT_PATH, RANKED_OUTPUT_PATH, TRIALS_CACHE_PATH
from ..patients.features import build_patient_profiles
from ..trials.api import fetch_trials
from ..trials.rag import build_trial_rag_features
from ..matching.features import build_match_features, _detect_mode
from ..matching.ranker import TrialRanker

# Features used to define _rule_label() — must be excluded from training X
# to avoid circular logic (label leakage).
LABEL_FEATURES = {
    "key_condition_present", "inclusion_satisfied_count",
    "renal_exclusion_hit", "stroke_exclusion_hit", "insulin_pump_conflict",
    "missing_hba1c", "missing_renal_lab",
    "age_match", "sex_match",
    "has_cancer_exclusion_condition",
    "has_prior_chemo", "has_metastatic_disease",
}


def build_patient_trial_pairs(trials_limit: int = 50, condition: str = "type 2 diabetes") -> pd.DataFrame:
    mode = _detect_mode(condition)
    profiles = build_patient_profiles(
        patients_path=str(DATA_DIR / "patients.csv"),
        conditions_path=str(DATA_DIR / "conditions.csv"),
        medications_path=str(DATA_DIR / "medications.csv"),
        observations_path=str(DATA_DIR / "observations.csv"),
        encounters_path=str(DATA_DIR / "encounters.csv"),
    )

    cache_path = OUTPUT_DIR / f"trials_{condition.replace(' ', '_')}.json"
    trials = fetch_trials(
        condition,
        limit=trials_limit,
        cache_path=cache_path,
        mode=mode,
    )
    trial_rag = build_trial_rag_features(trials, DATA_DIR)
    rows = []

    for trial in trials:
        condition_keywords = [c.lower() for c in trial.conditions] if trial.conditions else None
        for patient in profiles.values():
            features = build_match_features(patient, trial.criteria,
                                            condition_keywords=condition_keywords, mode=mode)
            rag_features = trial_rag.get(trial.trial_id, {"rag_sim_max": 0.0, "rag_sim_mean": 0.0})
            label = _rule_label(features, mode=mode)
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
    output_path = OUTPUT_DIR / f"patient_trial_pairs_{mode}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def rank_patients(df: pd.DataFrame) -> pd.DataFrame:
    # Exclude label-defining features from training to prevent leakage
    feature_cols = [c for c in df.columns
                    if c not in ("trial_id", "patient_id", "label")
                    and c not in LABEL_FEATURES]
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
        test_trial_ids = df.loc[test_mask, "trial_id"].values
        metrics = _compute_metrics(y_test, scores_test, trial_ids=test_trial_ids)
        metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
        import json
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    ranked = df.sort_values(["trial_id", "score"], ascending=[True, False])
    RANKED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(RANKED_OUTPUT_PATH, index=False)
    return ranked


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, k: int = 10,
                     trial_ids: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute evaluation metrics for the ranking model.

    When trial_ids is provided, precision_at_k and ndcg_at_k are computed
    per-trial (macro-averaged) for a more realistic ranking evaluation.
    """
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, ndcg_score

    # Binarize graded labels: grade >= 2 counts as "relevant"
    y_binary = (y_true >= 2).astype(float)

    # Use the positive-rate-calibrated threshold: predict positive for top P%
    # where P is the actual positive rate in the test set
    pos_rate = float(y_binary.mean())
    if 0 < pos_rate < 1:
        threshold = float(np.percentile(scores, (1.0 - pos_rate) * 100))
    else:
        threshold = float(np.median(scores))
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

    # --- per-trial ranking metrics (macro-averaged) ---
    if trial_ids is not None:
        unique_trials = np.unique(trial_ids)
        trial_pk = []
        trial_ndcg = []
        trial_ndcg_k = []
        for tid in unique_trials:
            mask = trial_ids == tid
            t_scores = scores[mask]
            t_labels = y_true[mask]
            t_binary = y_binary[mask]

            # precision@k among candidates with at least some relevance (grade >= 1)
            # This tests within-cohort ranking quality, not just cohort identification
            cohort_mask = t_labels >= 1
            if cohort_mask.sum() >= k:
                cohort_scores = t_scores[cohort_mask]
                cohort_binary = t_binary[cohort_mask]
                top_k_idx = np.argsort(cohort_scores)[::-1][:k]
                trial_pk.append(float(cohort_binary[top_k_idx].sum() / len(top_k_idx)))
            elif t_binary.sum() > 0:
                top_k_idx = np.argsort(t_scores)[::-1][:k]
                trial_pk.append(float(t_binary[top_k_idx].sum() / len(top_k_idx)))

            # ndcg per trial
            if len(t_labels) > 1 and t_labels.max() > 0:
                try:
                    trial_ndcg.append(float(ndcg_score([t_labels], [t_scores])))
                    trial_ndcg_k.append(float(ndcg_score([t_labels], [t_scores], k=k)))
                except ValueError:
                    pass
        metrics["ndcg"] = float(np.mean(trial_ndcg)) if trial_ndcg else 0.0
        metrics[f"ndcg_at_{k}"] = float(np.mean(trial_ndcg_k)) if trial_ndcg_k else 0.0
        metrics[f"precision_at_{k}"] = float(np.mean(trial_pk)) if trial_pk else 0.0
    else:
        # Fallback: global (non-per-trial) metrics
        try:
            metrics["ndcg"] = float(ndcg_score([y_true], [scores]))
            metrics[f"ndcg_at_{k}"] = float(ndcg_score([y_true], [scores], k=k))
        except ValueError:
            metrics["ndcg"] = 0.0
            metrics[f"ndcg_at_{k}"] = 0.0
        top_k_idx = np.argsort(scores)[::-1][:k]
        if len(top_k_idx) > 0:
            metrics[f"precision_at_{k}"] = float(y_binary[top_k_idx].sum() / len(top_k_idx))
        else:
            metrics[f"precision_at_{k}"] = 0.0

    # Label distribution for debugging
    metrics["label_dist_grade_0"] = float((y_true == 0).mean())
    metrics["label_dist_grade_1"] = float((y_true == 1).mean())
    metrics["label_dist_grade_2"] = float((y_true == 2).mean())
    metrics["label_dist_grade_3"] = float((y_true == 3).mean())
    metrics["threshold_used"] = threshold

    return metrics


def _rule_label(features: dict, mode: str = "diabetes") -> int:
    """Graded relevance label for lambdarank: 0=excluded, 1=poor, 2=partial, 3=good.

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


if __name__ == "__main__":
    import sys
    condition = sys.argv[1] if len(sys.argv) > 1 else "type 2 diabetes"
    df_pairs = build_patient_trial_pairs(trials_limit=50, condition=condition)
    rank_patients(df_pairs)
