from typing import Dict, List, Optional

from ..patients.features import PatientProfile
from ..trials.schema import TrialCriteria


# --------------- ranker mode configuration ---------------

RANKER_MODES = {
    "diabetes": {
        "condition_keywords": [
            "type 2 diabetes", "type ii diabetes", "t2d",
            "diabetes mellitus",
        ],
        "key_labs": [
            "a1c", "glucose", "creatinine", "cholesterol",
            "triglycerides", "hemoglobin",
        ],
    },
    "cancer": {
        "condition_keywords": [
            "cancer", "carcinoma", "neoplasm", "tumor", "tumour",
            "malignant", "lymphoma", "leukemia", "melanoma", "sarcoma",
            "oncology",
        ],
        "key_labs": [
            "hemoglobin", "white blood cell", "platelet",
            "creatinine", "albumin", "calcium",
        ],
    },
}

# Chemo-related medication keywords for cancer mode
_CHEMO_KEYWORDS = [
    "cyclophosphamide", "doxorubicin", "paclitaxel", "cisplatin",
    "carboplatin", "fluorouracil", "methotrexate", "vincristine",
    "irinotecan", "oxaliplatin", "gemcitabine", "docetaxel",
    "etoposide", "bleomycin", "capecitabine", "pemetrexed",
    "temozolomide", "nivolumab", "pembrolizumab", "atezolizumab",
    "trastuzumab", "bevacizumab", "rituximab", "cetuximab",
]


def _detect_mode(condition: str) -> str:
    """Detect ranker mode from condition string."""
    lower = condition.lower()
    for kw in RANKER_MODES["cancer"]["condition_keywords"]:
        if kw in lower:
            return "cancer"
    return "diabetes"


def build_match_features(
    patient: PatientProfile,
    criteria: TrialCriteria,
    condition_keywords: Optional[List[str]] = None,
    mode: str = "diabetes",
) -> Dict[str, float]:
    mode_cfg = RANKER_MODES.get(mode, RANKER_MODES["diabetes"])
    # Combine trial-specific keywords with mode-level keywords for robust matching
    base_kw = mode_cfg["condition_keywords"]
    if condition_keywords:
        kw_list = list(set(base_kw + condition_keywords))
    else:
        kw_list = base_kw

    features: Dict[str, float] = {}

    # --- binary match features ---
    features["age_match"] = _match_age(patient.age, criteria)
    features["sex_match"] = _match_sex(patient.sex, criteria)

    # Generic key condition — checks trial's actual target condition
    # Bidirectional substring match to handle different naming conventions
    # (e.g. "Diabetes Mellitus, Type 2" vs "type 2 diabetes")
    features["key_condition_present"] = (
        1.0 if any(
            any(kw in c or c in kw for c in patient.conditions)
            for kw in kw_list
        ) else 0.0
    )

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
    inclusion_fields = ["age_match", "sex_match", "key_condition_present", "hba1c_above_min", "hba1c_below_max"]
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
        features["diagnosis_overlap_score"] = features["key_condition_present"]

    # --- continuous / tie-breaking features ---
    features["age_normalized"] = patient.age / 100.0
    hba1c_val = _lab_value(patient, "a1c")
    features["hba1c_value"] = hba1c_val / 20.0 if hba1c_val is not None else 0.0
    features["medication_count"] = min(len(patient.medications), 30) / 30.0
    features["condition_count"] = min(len(patient.conditions), 30) / 30.0
    features["insulin_on_med"] = 1.0 if any("insulin" in m for m in patient.medications) else 0.0

    # Lab completeness: fraction of mode-relevant key labs present
    key_labs = mode_cfg["key_labs"]
    present = sum(1.0 for k in key_labs if _lab_value(patient, k) is not None)
    features["lab_completeness"] = present / len(key_labs)

    # --- cancer-specific features ---
    if mode == "cancer":
        features["has_prior_chemo"] = (
            1.0 if any(
                any(ck in m for ck in _CHEMO_KEYWORDS)
                for m in patient.medications
            ) else 0.0
        )
        features["has_prior_radiation"] = (
            1.0 if any("radiation" in c for c in patient.conditions)
            or any("radiation" in m for m in patient.medications)
            else 0.0
        )
        features["has_metastatic_disease"] = (
            1.0 if any(
                kw in c
                for c in patient.conditions
                for kw in ("metastatic", "metastasis", "stage iv", "stage 4")
            ) else 0.0
        )
        cancer_conds = sum(
            1 for c in patient.conditions
            if any(kw in c for kw in RANKER_MODES["cancer"]["condition_keywords"])
        )
        features["tumor_condition_count"] = min(cancer_conds, 10) / 10.0
        chemo_meds = sum(
            1 for m in patient.medications
            if any(ck in m for ck in _CHEMO_KEYWORDS)
        )
        features["cancer_medication_count"] = min(chemo_meds, 10) / 10.0

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
