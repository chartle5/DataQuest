"""Tests for parser fix: age regex must not match BMI, HbA1c, or duration ranges."""
from src.trials.parser import parse_eligibility_text


def test_bmi_not_parsed_as_age():
    text = "BMI is 19 to 30 kg/m2 and body weight >=50 kg"
    c = parse_eligibility_text(text)
    assert c.min_age is None
    assert c.max_age is None


def test_hba1c_range_not_parsed_as_age():
    text = "HbA1c: 7.0-11.0 %"
    c = parse_eligibility_text(text)
    assert c.min_age is None
    assert c.max_age is None


def test_duration_not_parsed_as_age():
    text = "visited the centre at least once in the last 3-6 months"
    c = parse_eligibility_text(text)
    assert c.min_age is None
    assert c.max_age is None


def test_real_age_still_parses():
    text = "Inclusion: adults 18-65 years with HbA1c >= 7.5%"
    c = parse_eligibility_text(text)
    assert c.min_age == 18
    assert c.max_age == 65


def test_age_gte_parses():
    text = "Age >= 65 years old"
    c = parse_eligibility_text(text)
    assert c.min_age == 65


def test_unicode_age_gte_parses():
    c = parse_eligibility_text("Age \u2265 18 years")
    assert c.min_age == 18
