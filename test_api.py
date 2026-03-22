import requests, json
r = requests.post('http://localhost:8080/api/rank', json={'condition':'cancer','top_n':9}, timeout=300)
print(f"Status: {r.status_code}")
if r.status_code != 200:
    print(f"Response: {r.text[:500]}")
else:
    data = r.json()
    print(f"Trial: {data.get('trial_id')}")
    print(f"Title: {data.get('trial_title')}")
    for i, c in enumerate(data.get('candidates', [])):
        reasons = '; '.join(c['reasons'])
        print(f"  {i+1}. {c['first_name']} {c['last_name']} | {c['confidence_score']}% {c['status']} | {reasons}")
