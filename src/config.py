from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "syntheaCSV-20260321T160919Z-3-001" / "syntheaCSV"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_CACHE_PATH = OUTPUT_DIR / "trials_diabetes.json"
PAIRS_OUTPUT_PATH = OUTPUT_DIR / "patient_trial_pairs.csv"
RANKED_OUTPUT_PATH = OUTPUT_DIR / "trial_ranked_patients.csv"
