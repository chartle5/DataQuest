"""Quick API test."""
import requests, json

for cond in ["type 2 diabetes", "cancer"]:
    print(f"\n=== {cond.upper()} (top 5) ===")
    r = requests.post("http://localhost:8080/api/rank", json={"condition": cond, "top_n": 5})
    data = r.json()
    print(f"Trial: {data.get('trial_id', '?')}")
    for i, c in enumerate(data.get("candidates", []), 1):
        print(f"  {i}. {c['first_name']} {c['last_name']}: {c['confidence_score']}% [{c['status']}]")
        for reason in c.get("reasons", []):
            print(f"     - {reason}")
