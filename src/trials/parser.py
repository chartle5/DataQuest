import re
from typing import Iterable, Optional, Tuple

from .schema import TrialCriteria
from .ner import extract_entities


# Age-range regex requires an age unit (years/yrs/year/y/year-old) to avoid
# false positives from BMI ranges ("19 to 30 kg/m2"), HbA1c ("7.0-11.0 %"),
# and duration phrases ("3-6 months").
AGE_RANGE_RE = re.compile(
    r"(?:aged?\s+)?(\d{1,3})\s*(?:to|-|–)\s*(\d{1,3})\s*(?:years|yrs|year|y(?:ears)?)[\s\-]?(?:old|of\s+age)?",
    re.IGNORECASE,
)
AGE_MIN_RE = re.compile(
    r"(?:>=|≥|at\s+least|older\s+than|\bmin(?:imum)?\b|age\s*>\s*=?)\s*(\d{1,3})\s*(?:years|yrs|year|y(?:ears)?)",
    re.IGNORECASE,
)
AGE_MAX_RE = re.compile(
    r"(?:<=|≤|at\s+most|younger\s+than|\bmax(?:imum)?\b|age\s*<\s*=?)\s*(\d{1,3})\s*(?:years|yrs|year|y(?:ears)?)",
    re.IGNORECASE,
)
HBA1C_RE = re.compile(r"hba1c[^\d]*(>=|>|<=|<)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


INCLUSION_MARKERS = ["inclusion criteria", "inclusion"]
EXCLUSION_MARKERS = ["exclusion criteria", "exclusion"]

DIABETES_TERMS = ["type 2 diabetes", "type ii diabetes", "t2d", "t2dm", "diabetes mellitus type 2"]
RENAL_FAILURE_TERMS = ["renal failure", "end stage renal", "end-stage renal", "dialysis", "ckd stage 4", "ckd stage 5"]
STROKE_TERMS = ["stroke", "cva", "cerebrovascular accident"]
INSULIN_PUMP_TERMS = ["insulin pump", "csii"]
INSULIN_EXCLUDE_TERMS = ["no insulin", "not on insulin", "insulin-naive", "insulin naive"]


def _split_criteria(text: str) -> Tuple[str, str]:
    lower = text.lower()
    inclusion_text = text
    exclusion_text = ""

    for marker in EXCLUSION_MARKERS:
        idx = lower.find(marker)
        if idx >= 0:
            inclusion_text = text[:idx]
            exclusion_text = text[idx:]
            return inclusion_text, exclusion_text

    return inclusion_text, exclusion_text


def _parse_age(text: str) -> Tuple[Optional[int], Optional[int]]:
    m = AGE_RANGE_RE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        # Sanity: reject ranges that are obviously not human ages
        if 1 <= lo <= 120 and 1 <= hi <= 120 and lo < hi:
            return lo, hi

    min_age = None
    max_age = None
    m = AGE_MIN_RE.search(text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 120:
            min_age = val
    m = AGE_MAX_RE.search(text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 120:
            max_age = val

    return min_age, max_age


# Reproductive / contraception context keywords — standard clinical trial
# boilerplate that mentions "female" without restricting the trial to women.
_REPRODUCTIVE_CONTEXT_KEYWORDS = [
    "childbearing", "contracepti", "pregnan", "breastfeed",
    "nursing", "fertile", "menstruat", "birth control",
    "reproductive potential", "negative pregnancy",
]


def _female_only_in_reproductive_context(text: str) -> bool:
    """Return True if every 'female' mention is near reproductive context words."""
    for m in re.finditer(r"\bfemale\b", text):
        start = max(0, m.start() - 120)
        end = min(len(text), m.end() + 120)
        window = text[start:end]
        if not any(kw in window for kw in _REPRODUCTIVE_CONTEXT_KEYWORDS):
            return False
    return True


def _parse_sex(text: str) -> Optional[str]:
    lower = text.lower()
    has_female = re.search(r"\bfemale\b", lower) is not None
    has_male = re.search(r"\bmale\b", lower) is not None
    if has_female and has_male:
        return "all"
    if has_female:
        # If "female" only appears near reproductive/contraception language,
        # the trial is open to all sexes (standard boilerplate).
        if _female_only_in_reproductive_context(lower):
            return "all"
        return "female"
    if has_male:
        return "male"
    return None


def _parse_hba1c(text: str) -> Tuple[Optional[float], Optional[float]]:
    min_val = None
    max_val = None
    for match in HBA1C_RE.finditer(text):
        op = match.group(1)
        value = float(match.group(2))
        if op in (">=", ">"):
            min_val = value if min_val is None else max(min_val, value)
        elif op in ("<=", "<"):
            max_val = value if max_val is None else min(max_val, value)
        else:
            min_val = min_val or value
    return min_val, max_val


def parse_eligibility_text(text: str) -> TrialCriteria:
    inclusion_text, exclusion_text = _split_criteria(text)
    min_age, max_age = _parse_age(text)
    sex_allowed = _parse_sex(text)
    hba1c_min, hba1c_max = _parse_hba1c(text)

    lower_text = text.lower()
    requires_t2d = _contains_any(inclusion_text, DIABETES_TERMS)
    excludes_renal_failure = _contains_any(exclusion_text, RENAL_FAILURE_TERMS)
    excludes_recent_stroke = _contains_any(exclusion_text, STROKE_TERMS) and _has_recent_language(exclusion_text)
    excludes_insulin_pump = _contains_any(exclusion_text, INSULIN_PUMP_TERMS)
    allows_insulin = _infer_insulin_allowance(text)

    entities = [e.lower() for e in extract_entities(text)]
    if not requires_t2d and _contains_any(" ".join(entities), DIABETES_TERMS):
        requires_t2d = True
    if not excludes_renal_failure and _contains_any(" ".join(entities), RENAL_FAILURE_TERMS):
        excludes_renal_failure = True

    return TrialCriteria(
        min_age=min_age,
        max_age=max_age,
        sex_allowed=sex_allowed,
        requires_t2d=requires_t2d,
        hba1c_min=hba1c_min,
        hba1c_max=hba1c_max,
        excludes_renal_failure=excludes_renal_failure,
        excludes_recent_stroke=excludes_recent_stroke,
        excludes_insulin_pump=excludes_insulin_pump,
        allows_insulin=allows_insulin,
        entities=sorted(set(entities)),
        raw_inclusion_text=inclusion_text.strip() or None,
        raw_exclusion_text=exclusion_text.strip() or None,
    )


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(term in lower for term in terms)


def _has_recent_language(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ["within", "last", "past", "recent"])


def _infer_insulin_allowance(text: str) -> Optional[bool]:
    lower = text.lower()
    if "insulin" not in lower:
        return None
    if _contains_any(lower, INSULIN_EXCLUDE_TERMS):
        return False
    return True
