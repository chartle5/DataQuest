# Clinical Trial Pre-screening — Candidate Ranking Prototype

This repository contains a prototype pipeline to rank patients for clinical trials (pre-screening). It is a decision-support tool only — not a final eligibility engine.

Overview
- Data loading and cleaning from local CSVs in `data/`.
- Feature engineering to create patient–trial pair features.
- Simple ranking service to return top-N candidates for a trial with human-readable rationale.

Quickstart
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the basic preprocessing to create `data/processed/patients.parquet`:
```bash
python -c "from src.data.loader import process_all; process_all()"
```
3. Example usage from Python:
```py
from src.trials.api import load_trials_json
from src.ranking.service import get_top_candidates
trials = load_trials_json('data/trials_sample.json')
trial = trials[0]
df = get_top_candidates(trial, n=10)
print(df.head())
```

Running the server (for frontend integration)

1. Start the Flask server (serves `/api/rank` and `/api/train`):
```bash
python -m src.server
```

2. Serve the frontend directory during development (from project root):
```bash
python -m http.server 8000 --directory frontend
```

Open `http://localhost:8000` in your browser. The frontend calls the server at `/api/rank`.

Training models

To train a model for a condition (e.g., `type 2 diabetes`):
```bash
python -c "from src.matching.train_models import train_for_condition; print(train_for_condition('type 2 diabetes'))"
```

This will produce a model file under `outputs/models/type_2_diabetes_model.pkl`.

Testing end-to-end

- Start the Flask server and static frontend as above.
- In the UI choose `Type II Diabetes` or `Cancer` and request Top N candidates.
- Alternatively, call the API directly:
```bash
curl -X POST localhost:8080/api/rank -H 'Content-Type: application/json' -d '{"condition":"type 2 diabetes","top_n":5}'
```

Notes & Limitations
- If model artifacts are not present the server will use a heuristic scorer. Use `/api/train` to train a model (may take time).
- The training pipeline uses rule-based pseudo-labeling; inspect `src/matching/train_models.py` for label logic.

Notes
- This is an initial scaffold. Next steps: implement robust feature engineering, weak supervision labels, model training (LightGBM), and frontend wiring.

Privacy
- The pipeline masks or avoids exporting direct identifiers (SSNs) in outputs. Review the `src/data/loader.py` behavior before sharing.
