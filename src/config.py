from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Default expected Synthea export path (may not exist in every workspace)
_SYNTHEA_PATH = PROJECT_ROOT / "data" / "syntheaCSV-20260321T160919Z-3-001" / "syntheaCSV"
# Fallback to repository-level data/ when the Synthea path is not present
if _SYNTHEA_PATH.exists() and (_SYNTHEA_PATH / "patients.csv").exists():
	DATA_DIR = _SYNTHEA_PATH
else:
	# Use the top-level data/ directory that is included in this repo
	DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_CACHE_PATH = OUTPUT_DIR / "trials_diabetes.json"
PAIRS_OUTPUT_PATH = OUTPUT_DIR / "patient_trial_pairs.csv"
RANKED_OUTPUT_PATH = OUTPUT_DIR / "trial_ranked_patients.csv"
