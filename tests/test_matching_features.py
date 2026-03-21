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
