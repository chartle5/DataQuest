from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd


@dataclass
class PatientProfile:
    patient_id: str
    age: int
    sex: str
    conditions: Set[str] = field(default_factory=set)
    medications: Set[str] = field(default_factory=set)
    labs: Dict[str, float] = field(default_factory=dict)
    recent_hospitalization: bool = False


def _calc_age(birthdate: str, as_of: Optional[datetime]) -> int:
    if as_of is None:
        as_of = datetime.utcnow()
    dob = datetime.fromisoformat(birthdate)
    return as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))


def build_patient_profiles(
    patients_path: str,
    conditions_path: str,
    medications_path: str,
    observations_path: str,
    encounters_path: str,
    as_of: Optional[datetime] = None,
) -> Dict[str, PatientProfile]:
    patients = pd.read_csv(patients_path)
    profiles: Dict[str, PatientProfile] = {}

    for _, row in patients.iterrows():
        patient_id = row["Id"]
        age = _calc_age(row["BIRTHDATE"], as_of)
        sex = row["GENDER"].lower()
        profiles[patient_id] = PatientProfile(patient_id=patient_id, age=age, sex=sex)

    conditions = pd.read_csv(conditions_path)
    for _, row in conditions.iterrows():
        patient_id = row["PATIENT"]
        desc = str(row["DESCRIPTION"]).lower()
        if patient_id in profiles:
            profiles[patient_id].conditions.add(desc)

    meds = pd.read_csv(medications_path)
    for _, row in meds.iterrows():
        patient_id = row["PATIENT"]
        desc = str(row["DESCRIPTION"]).lower()
        if patient_id in profiles:
            profiles[patient_id].medications.add(desc)

    obs = pd.read_csv(observations_path)
    obs = obs.sort_values("DATE")
    for _, row in obs.iterrows():
        patient_id = row["PATIENT"]
        desc = str(row["DESCRIPTION"]).lower()
        value = row["VALUE"]
        if patient_id in profiles and _is_number(value):
            profiles[patient_id].labs[desc] = float(value)

    encounters = pd.read_csv(encounters_path)
    for _, row in encounters.iterrows():
        patient_id = row["PATIENT"]
        enc_class = str(row.get("ENCOUNTERCLASS", "")).lower()
        if patient_id in profiles and enc_class == "inpatient":
            profiles[patient_id].recent_hospitalization = True

    return profiles


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
