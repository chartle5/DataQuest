from src.trials.parser import parse_eligibility_text


def test_parse_age_and_hba1c():
    text = "Inclusion: adults 18-65 years with HbA1c >= 7.5%"
    criteria = parse_eligibility_text(text)
    assert criteria.min_age == 18
    assert criteria.max_age == 65
    assert criteria.hba1c_min == 7.5


def test_parse_sex():
    text = "Inclusion: female participants"
    criteria = parse_eligibility_text(text)
    assert criteria.sex_allowed == "female"


def test_exclusion_scoping():
    text = "Inclusion: adults with type 2 diabetes. Exclusion: renal failure or stroke within 6 months."
    criteria = parse_eligibility_text(text)
    assert criteria.excludes_renal_failure is True
    assert criteria.excludes_recent_stroke is True


def test_insulin_allowance():
    text = "Exclusion: no insulin use"
    criteria = parse_eligibility_text(text)
    assert criteria.allows_insulin is False
