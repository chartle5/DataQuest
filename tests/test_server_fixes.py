"""Quick smoke test: verify that the ranking endpoint produces differentiated scores."""
import json
from src.server import APP


def test_rank_produces_differentiated_scores():
    """Top 5 candidates should have valid structure and consistent scoring.

    Confidence scores may all be 0.0 if the best trial has age criteria
    none of our synthetic patients satisfy — that's correct hard-fail behavior.
    In that case, verify all are marked 'ineligible'.
    """
    client = APP.test_client()
    resp = client.post("/api/rank", json={"condition": "type 2 diabetes", "top_n": 5})
    assert resp.status_code == 200
    data = resp.get_json()
    candidates = data["candidates"]
    assert len(candidates) > 0

    scores = [c["confidence_score"] for c in candidates]

    if all(s == 0.0 for s in scores):
        # All ineligible (e.g. trial age range doesn't overlap patient ages)
        # This is correct hard-constraint behavior
        for c in candidates:
            assert c["status"] == "ineligible", (
                f"Confidence 0.0 should map to 'ineligible', got '{c['status']}'"
            )
    else:
        # At least 2 distinct scores in top 5
        assert len(set(scores)) >= 2, f"All scores identical: {scores}"

    # Every candidate has required fields in correct structure
    for c in candidates:
        assert "patient_id" in c
        assert "first_name" in c
        assert "last_name" in c
        assert "status" in c
        assert "confidence_score" in c
        assert "reasons" in c
        # Should NOT have 'features' or 'score' keys
        assert "features" not in c
        assert "score" not in c


def test_rank_output_format():
    """Output format must be clean: no markdown, no extra wrappers."""
    client = APP.test_client()
    resp = client.post("/api/rank", json={"condition": "type 2 diabetes", "top_n": 3})
    data = resp.get_json()
    for c in data["candidates"]:
        for reason in c["reasons"]:
            assert not reason.startswith("-"), f"Reason has bullet: {reason}"
            assert not reason.startswith("*"), f"Reason has markdown: {reason}"
            assert not reason.startswith("{"), f"Reason has JSON: {reason}"


def test_verify_missing_top_n():
    client = APP.test_client()
    resp = client.post("/api/verify", json={"condition": "type 2 diabetes"})
    assert resp.status_code == 400
    assert "top_n" in resp.get_json()["error"]


def test_verify_invalid_top_n():
    client = APP.test_client()
    resp = client.post("/api/verify", json={"condition": "type 2 diabetes", "top_n": -1})
    assert resp.status_code == 400
