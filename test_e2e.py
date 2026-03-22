"""End-to-end test of cancer and diabetes ranking after fixes."""
from src.server import APP

print("=== CANCER RANKING TEST ===")
with APP.test_client() as client:
    r = client.post("/api/rank", json={"condition": "cancer", "top_n": 5})
    data = r.get_json()

print(f"Trial: {data['trial_id']}")
print(f"Title: {data['trial_title']}")
for i, c in enumerate(data["candidates"], 1):
    print(f"\n  {i}. {c['first_name']} {c['last_name']} ({c['patient_id'][:20]}...)")
    print(f"     Confidence: {c['confidence_score']}% [{c['status']}]")
    for reason in c.get("reasons", []):
        print(f"     - {reason}")

print("\n\n=== DIABETES RANKING TEST ===")
with APP.test_client() as client:
    r = client.post("/api/rank", json={"condition": "type 2 diabetes", "top_n": 5})
    data = r.get_json()

print(f"Trial: {data['trial_id']}")
print(f"Title: {data['trial_title']}")
for i, c in enumerate(data["candidates"], 1):
    print(f"\n  {i}. {c['first_name']} {c['last_name']} ({c['patient_id'][:20]}...)")
    print(f"     Confidence: {c['confidence_score']}% [{c['status']}]")
    for reason in c.get("reasons", []):
        print(f"     - {reason}")

print("\n\n=== CANCER METRICS ===")
with APP.test_client() as client:
    r = client.get("/api/metrics?mode=cancer")
    import json
    print(json.dumps(r.get_json(), indent=2))

print("\n=== DIABETES METRICS ===")
with APP.test_client() as client:
    r = client.get("/api/metrics?mode=diabetes")
    print(json.dumps(r.get_json(), indent=2))
