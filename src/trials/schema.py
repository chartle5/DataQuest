from typing import List, Optional
from pydantic import BaseModel


class TrialCriteria(BaseModel):
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    sex_allowed: Optional[str] = None  # "all", "male", "female"
    requires_t2d: bool = False
    hba1c_min: Optional[float] = None
    hba1c_max: Optional[float] = None
    excludes_renal_failure: bool = False
    excludes_recent_stroke: bool = False
    excludes_insulin_pump: bool = False
    allows_insulin: Optional[bool] = None
    entities: List[str] = []
    raw_inclusion_text: Optional[str] = None
    raw_exclusion_text: Optional[str] = None


class TrialRecord(BaseModel):
    trial_id: str
    title: str
    conditions: List[str] = []
    eligibility_text: str = ""
    criteria: TrialCriteria
