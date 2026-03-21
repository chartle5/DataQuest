"""Simple feature engineering for patient-trial pair features."""
from typing import Dict
import pandas as pd


def compute_pair_features(patients: pd.DataFrame, trial: Dict) -> pd.DataFrame:
    """Given a patients DataFrame and a trial record (dict-like), return a DataFrame
    with basic pairwise features and simple rationale notes.

    Expected trial keys: min_age, max_age, sex (M/F/All), conditions (list)
    """
    df = patients.copy()

    min_age = trial.get("min_age")
    max_age = trial.get("max_age")
    sex_req = (trial.get("sex_eligibility") or trial.get("sex") or "All").upper()

    # Age features
    if min_age is not None and max_age is not None:
        df["age_in_range"] = df["age"].between(min_age, max_age)
        df["age_distance"] = df["age"].apply(lambda x: 0 if pd.isna(x) else max(0, min_age - x, x - max_age))
    else:
        df["age_in_range"] = True
        df["age_distance"] = 0

    # Sex match
    if sex_req in ("M", "F"):
        df["sex_match"] = df["gender"] == sex_req
    else:
        df["sex_match"] = True

    # Condition match: check type2 flag or keywords
    cond_keywords = [c.lower() for c in (trial.get("conditions") or [])]
    if cond_keywords:
        df["condition_match"] = df["type2_diabetes_flag"]
    else:
        df["condition_match"] = False

    # completeness score
    req_cols = ["age", "gender", "type2_diabetes_flag"]
    df["completeness_score"] = df[req_cols].notnull().sum(axis=1) / len(req_cols)

    return df
