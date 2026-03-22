"""Ranking service to return top N candidates for a trial.

This is a simple heuristic-based scorer for initial prototype. Later this will call
the trained model and use SHAP for explainability.
"""
from typing import Dict, Any
import pandas as pd
from pathlib import Path

from src.features.engineer import compute_pair_features


PROCESSED = Path("data/processed")


def get_top_candidates(trial: Dict[str, Any], n: int = 10) -> pd.DataFrame:
    """Return top-N candidate patients for the given trial record.

    trial: mapping with keys like 'min_age', 'max_age', 'sex_eligibility', 'conditions'
    """
    patients_path = PROCESSED / "patients.parquet"
    if not patients_path.exists():
        raise FileNotFoundError("Processed patients not found. Run src.data.loader.process_all() first.")

    patients = pd.read_parquet(patients_path)
    df = compute_pair_features(patients, trial)

    # build a simple heuristic score
    # weight age_in_range (0.4), condition_match (0.4), completeness (0.2)
    df["score"] = (
        df["age_in_range"].astype(int) * 0.4 + df["condition_match"].astype(int) * 0.4 + df["completeness_score"] * 0.2
    )

    # sort and select top N
    df_sorted = df.sort_values("score", ascending=False).head(n)

    # add rationale strings
    rationale = []
    for _, row in df_sorted.iterrows():
        parts = []
        if row.get("age_in_range"):
            parts.append("age within required range")
        else:
            parts.append(f"age outside range (distance={row.get('age_distance')})")
        if row.get("condition_match"):
            parts.append("condition matches trial (Type 2 Diabetes)")
        else:
            parts.append("condition not matched")
        parts.append(f"completeness={row.get('completeness_score'):.2f}")
        rationale.append("; ".join(parts))

    df_sorted = df_sorted.assign(rationale=rationale)

    # Select output columns
    out_cols = ["id", "first", "last", "score", "rationale"]
    available = [c for c in out_cols if c in df_sorted.columns]
    return df_sorted[available]


if __name__ == "__main__":
    import json
    sample_trial = {"min_age": 18, "max_age": 75, "conditions": ["Type 2 Diabetes"], "sex_eligibility": "All"}
    print(get_top_candidates(sample_trial, n=5).head())
