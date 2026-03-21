from src.matching.features import build_match_features
from src.patients.features import PatientProfile
from src.trials.schema import TrialCriteria


def test_match_features_basic():
    patient = PatientProfile(
        patient_id="p1",
        age=50,
        sex="male",
        conditions={"type 2 diabetes"},
        medications=set(),
        labs={"hba1c": 8.0},
        recent_hospitalization=False,
    )
    criteria = TrialCriteria(min_age=18, max_age=65, hba1c_min=7.5, requires_t2d=True)
    features = build_match_features(patient, criteria)
    assert features["age_match"] == 1.0
    assert features["hba1c_above_min"] == 1.0
    assert features["age_gap"] == 0.0
    assert features["inclusion_satisfied_count"] >= 3.0


def test_match_features_exclusions():
    patient = PatientProfile(
        patient_id="p2",
        age=50,
        sex="female",
        conditions={"type 2 diabetes", "renal failure"},
        medications=set(),
        labs={"hba1c": 8.0},
        recent_hospitalization=False,
    )
    criteria = TrialCriteria(min_age=18, max_age=65, requires_t2d=True)
    features = build_match_features(patient, criteria)
    assert features["renal_exclusion_hit"] == 1.0
    assert features["exclusion_conflict_count"] >= 1.0


def test_match_features_missing_labs():
    patient = PatientProfile(
        patient_id="p3",
        age=30,
        sex="male",
        conditions={"type 2 diabetes"},
        medications=set(),
        labs={},
        recent_hospitalization=False,
    )
    criteria = TrialCriteria(min_age=18, max_age=65, hba1c_min=7.0, requires_t2d=True)
    features = build_match_features(patient, criteria)
    assert features["missing_hba1c"] == 1.0
    assert features["unknown_field_count"] >= 1.0
    assert features["hba1c_gap"] == 0.5  # uncertainty penalty


def test_match_features_age_gap():
    patient = PatientProfile(
        patient_id="p4",
        age=80,
        sex="male",
        conditions=set(),
        medications=set(),
        labs={},
        recent_hospitalization=False,
    )
    criteria = TrialCriteria(min_age=18, max_age=65)
    features = build_match_features(patient, criteria)
    assert features["age_match"] == 0.0
    assert features["age_gap"] > 0.0
