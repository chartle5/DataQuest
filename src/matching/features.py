from typing import Dict, Optional

from ..patients.features import PatientProfile
from ..trials.schema import TrialCriteria


def build_match_features(patient: PatientProfile, criteria: TrialCriteria) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # --- binary match features ---
    features["age_match"] = _match_age(patient.age, criteria)
    features["sex_match"] = _match_sex(patient.sex, criteria)
    features["t2d_present"] = _has_condition(patient, "type 2 diabetes")
    features["hba1c_above_min"] = _lab_meets_min(patient, "a1c", criteria.hba1c_min)
    features["hba1c_below_max"] = _lab_meets_max(patient, "a1c", criteria.hba1c_max)
    features["renal_exclusion_hit"] = _has_condition(patient, "renal failure")
    features["stroke_exclusion_hit"] = _has_condition(patient, "stroke")
    features["insulin_pump_conflict"] = _has_medication(patient, "insulin pump")

    features["missing_hba1c"] = 1.0 if _lab_missing(patient, "a1c") else 0.0
    features["missing_renal_lab"] = 1.0 if _lab_missing(patient, "creatinine") else 0.0
    features["recent_hospitalization"] = 1.0 if patient.recent_hospitalization else 0.0

    # --- normalized gap features ---
    features["age_gap"] = _normalized_age_gap(patient.age, criteria)
    features["hba1c_gap"] = _normalized_lab_gap(patient, "a1c", criteria.hba1c_min, criteria.hba1c_max)

    # --- aggregate features ---
    inclusion_fields = ["age_match", "sex_match", "t2d_present", "hba1c_above_min", "hba1c_below_max"]
    features["inclusion_satisfied_count"] = sum(features[f] for f in inclusion_fields)

    exclusion_fields = ["renal_exclusion_hit", "stroke_exclusion_hit", "insulin_pump_conflict"]
    features["exclusion_conflict_count"] = sum(features[f] for f in exclusion_fields)

    missing_fields = ["missing_hba1c", "missing_renal_lab"]
    features["unknown_field_count"] = sum(features[f] for f in missing_fields)

    # --- condition overlap ---
    trial_conditions = [c.lower() for c in (criteria.entities or [])]
    if trial_conditions:
        overlap = sum(1.0 for tc in trial_conditions if any(tc in pc for pc in patient.conditions))
        features["diagnosis_overlap_score"] = overlap / len(trial_conditions)
    else:
        features["diagnosis_overlap_score"] = features["t2d_present"]

    return features


def _match_age(age: int, criteria: TrialCriteria) -> float:
    if criteria.min_age is not None and age < criteria.min_age:
        return 0.0
    if criteria.max_age is not None and age > criteria.max_age:
        return 0.0
    return 1.0


def _normalized_age_gap(age: int, criteria: TrialCriteria) -> float:
    """0.0 when in range, positive when outside; normalized by range width."""
    min_a = criteria.min_age or 0
    max_a = criteria.max_age or 120
    width = max(max_a - min_a, 1)
    if age < min_a:
        return (min_a - age) / width
    if age > max_a:
        return (age - max_a) / width
    return 0.0


def _normalized_lab_gap(patient: PatientProfile, keyword: str,
                        threshold_min: Optional[float], threshold_max: Optional[float]) -> float:
    value = _lab_value(patient, keyword)
    if value is None:
        return 0.5  # uncertainty penalty
    if threshold_min is not None and value < threshold_min:
        return abs(threshold_min - value) / max(threshold_min, 1.0)
    if threshold_max is not None and value > threshold_max:
        return abs(value - threshold_max) / max(threshold_max, 1.0)
    return 0.0


def _match_sex(sex: str, criteria: TrialCriteria) -> float:
    if criteria.sex_allowed in (None, "all"):
        return 1.0
    return 1.0 if sex == criteria.sex_allowed else 0.0


def _has_condition(patient: PatientProfile, keyword: str) -> float:
    return 1.0 if any(keyword in c for c in patient.conditions) else 0.0


def _has_medication(patient: PatientProfile, keyword: str) -> float:
    return 1.0 if any(keyword in m for m in patient.medications) else 0.0


def _lab_meets_min(patient: PatientProfile, keyword: str, threshold: Optional[float]) -> float:
    if threshold is None:
        return 1.0
    value = _lab_value(patient, keyword)
    if value is None:
        return 0.0
    return 1.0 if value >= threshold else 0.0


def _lab_meets_max(patient: PatientProfile, keyword: str, threshold: Optional[float]) -> float:
    if threshold is None:
        return 1.0
    value = _lab_value(patient, keyword)
    if value is None:
        return 0.0
    return 1.0 if value <= threshold else 0.0


def _lab_missing(patient: PatientProfile, keyword: str) -> bool:
    return _lab_value(patient, keyword) is None


def _lab_value(patient: PatientProfile, keyword: str) -> Optional[float]:
    for desc, value in patient.labs.items():
        if keyword in desc:
            return value
    return None
