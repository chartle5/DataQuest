from dataclasses import asdict
from typing import Dict

from ..patients.features import PatientProfile
from ..trials.schema import TrialCriteria


def build_match_features(patient: PatientProfile, criteria: TrialCriteria) -> Dict[str, float]:
    features: Dict[str, float] = {}

    features["age_match"] = _match_age(patient.age, criteria)
    features["sex_match"] = _match_sex(patient.sex, criteria)
    features["t2d_present"] = _has_condition(patient, "type 2 diabetes")
    features["hba1c_above_min"] = _lab_meets_min(patient, "hba1c", criteria.hba1c_min)
    features["hba1c_below_max"] = _lab_meets_max(patient, "hba1c", criteria.hba1c_max)
    features["renal_exclusion_hit"] = _has_condition(patient, "renal failure")
    features["stroke_exclusion_hit"] = _has_condition(patient, "stroke")
    features["insulin_pump_conflict"] = _has_medication(patient, "insulin pump")

    features["missing_hba1c"] = 1.0 if _lab_missing(patient, "hba1c") else 0.0
    features["missing_renal_lab"] = 1.0 if _lab_missing(patient, "creatinine") else 0.0

    return features


def _match_age(age: int, criteria: TrialCriteria) -> float:
    if criteria.min_age is not None and age < criteria.min_age:
        return 0.0
    if criteria.max_age is not None and age > criteria.max_age:
        return 0.0
    return 1.0


def _match_sex(sex: str, criteria: TrialCriteria) -> float:
    if criteria.sex_allowed in (None, "all"):
        return 1.0
    return 1.0 if sex == criteria.sex_allowed else 0.0


def _has_condition(patient: PatientProfile, keyword: str) -> float:
    return 1.0 if any(keyword in c for c in patient.conditions) else 0.0


def _has_medication(patient: PatientProfile, keyword: str) -> float:
    return 1.0 if any(keyword in m for m in patient.medications) else 0.0


def _lab_meets_min(patient: PatientProfile, keyword: str, threshold: float | None) -> float:
    if threshold is None:
        return 1.0
    value = _lab_value(patient, keyword)
    if value is None:
        return 0.0
    return 1.0 if value >= threshold else 0.0


def _lab_meets_max(patient: PatientProfile, keyword: str, threshold: float | None) -> float:
    if threshold is None:
        return 1.0
    value = _lab_value(patient, keyword)
    if value is None:
        return 0.0
    return 1.0 if value <= threshold else 0.0


def _lab_missing(patient: PatientProfile, keyword: str) -> bool:
    return _lab_value(patient, keyword) is None


def _lab_value(patient: PatientProfile, keyword: str) -> float | None:
    for desc, value in patient.labs.items():
        if keyword in desc:
            return value
    return None
