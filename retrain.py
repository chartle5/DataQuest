"""Quick retraining script for both models."""
import sys
from src.matching.train_models import train_for_condition

conditions = sys.argv[1:] if len(sys.argv) > 1 else ["type 2 diabetes", "cancer"]
for cond in conditions:
    print(f"Training model for: {cond}")
    m = train_for_condition(cond, trials_limit=50)
    print(f"Model saved: {m}")
print("Done.")
