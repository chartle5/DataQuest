"""Diagnostic: understand label distribution, model score ranges, and confidence output."""
import json
import numpy as np
import pandas as pd
import joblib

from src.patients.features import build_patient_profiles
from src.trials.api import load_trials_json
from src.matching.features import build_match_features, _detect_mode, RANKER_MODES
from src import config

DATA_DIR = config.DATA_DIR

# 1. Check label distribution for diabetes training
print("=" * 60)
print("1. LABEL DISTRIBUTION FOR DIABETES")
print("=" * 60)
profiles = build_patient_profiles(
    patients_path=str(DATA_DIR / "patients.csv"),
    conditions_path=str(DATA_DIR / "conditions.csv"),
    medications_path=str(DATA_DIR / "medications.csv"),
    observations_path=str(DATA_DIR / "observations.csv"),
    encounters_path=str(DATA_DIR / "encounters.csv"),
)

trials = load_trials_json(str(config.OUTPUT_DIR / "trials_type_2_diabetes.json"))
trial = trials[0]  # Just check one trial
condition_keywords = [c.lower() for c in trial.conditions] if trial.conditions else None

from src.matching.train_models import _rule_label, LABEL_FEATURES

labels = []
features_list = []
for pid, patient in profiles.items():
    feat = build_match_features(patient, trial.criteria,
                                condition_keywords=condition_keywords, mode="diabetes")
    label = _rule_label(feat)
    labels.append(label)
    features_list.append(feat)

labels = np.array(labels)
print(f"Total patients: {len(labels)}")
for g in [0, 1, 2, 3]:
    pct = (labels == g).sum() / len(labels) * 100
    print(f"  Label {g}: {(labels == g).sum()} ({pct:.1f}%)")

binary = (labels >= 2).astype(int)
print(f"\nBinary relevant (label>=2): {binary.sum()} ({binary.sum()/len(binary)*100:.1f}%)")
print(f"Binary irrelevant (label<2): {(1-binary).sum()} ({(1-binary).sum()/len(binary)*100:.1f}%)")

# 2. Check model score range
print("\n" + "=" * 60)
print("2. MODEL SCORE DISTRIBUTION")
print("=" * 60)
model = joblib.load("outputs/models/type_2_diabetes_model.pkl")
print(f"Model type: {type(model).__name__}")
_mf = getattr(model, 'feature_names_in_', None)
if _mf is None or len(_mf) == 0:
    _mf = getattr(model, 'feature_name_', None)
model_features = list(_mf) if _mf is not None and len(_mf) > 0 else []
print(f"Model features ({len(model_features)}): {model_features}")

# Build features for all patients against this one trial
rows = []
for pid, patient in profiles.items():
    feat = build_match_features(patient, trial.criteria,
                                condition_keywords=condition_keywords, mode="diabetes")
    feat["rag_sim_max"] = 0.0
    feat["rag_sim_mean"] = 0.0
    rows.append({"patient_id": pid, **feat})

df = pd.DataFrame(rows)
if model_features:
    X = df[model_features].values.astype(float)
else:
    feature_cols = [c for c in df.columns if c != "patient_id"]
    X = df[feature_cols].values.astype(float)

scores = model.predict(X)
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")
print(f"Score percentiles: p10={np.percentile(scores,10):.4f}, p50={np.percentile(scores,50):.4f}, p90={np.percentile(scores,90):.4f}, p99={np.percentile(scores,99):.4f}")

# 3. Check confidence computation
print("\n" + "=" * 60)
print("3. CONFIDENCE ANALYSIS")
print("=" * 60)
# Sigmoid with current params: 1/(1+exp(-6*(score-0.3)))
for s in [scores.min(), np.percentile(scores,25), np.percentile(scores,50),
          np.percentile(scores,75), scores.max()]:
    scaled = 1.0 / (1.0 + np.exp(-6.0 * (s - 0.3)))
    print(f"  score={s:.4f} -> sigmoid={scaled*100:.1f}%")

# What % of top patients get 100% confidence?
df["score"] = scores
df_sorted = df.sort_values("score", ascending=False).head(10)
print(f"\nTop 10 scores: {df_sorted['score'].values}")

# Full confidence check for top 10
feature_cols = [c for c in df.columns if c not in ("patient_id", "score")]
for _, row in df_sorted.iterrows():
    feat = {c: float(row[c]) for c in feature_cols}
    raw_score = float(row["score"])
    # Reproduce confidence logic
    if feat.get("age_match", 1.0) == 0.0 or feat.get("sex_match", 1.0) == 0.0:
        conf = 0.0
    else:
        scaled = 1.0 / (1.0 + np.exp(-6.0 * (raw_score - 0.3)))
        conf = scaled * 100.0
        missing = feat.get("unknown_field_count", 0)
        conf -= missing * 8.0
        exclusions = feat.get("exclusion_conflict_count", 0)
        if exclusions > 0:
            conf = min(conf, 15.0)
        if feat.get("key_condition_present", 0.0) == 0.0:
            conf = min(conf, 15.0)
        conf = round(max(0.0, min(100.0, conf)), 2)
    print(f"  score={raw_score:.4f} -> confidence={conf}% (missing={feat.get('unknown_field_count',0)}, key_cond={feat.get('key_condition_present',0)})")
