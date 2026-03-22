"""Debug cancer matching pipeline - post-fix verification."""
import json
import joblib
import numpy as np

# Check the retrained model
print("--- Retrained model check ---")
model = joblib.load("outputs/models/cancer_model.pkl")
print(f"Model type: {type(model).__name__}")
if hasattr(model, 'feature_names_in_'):
    print(f"Features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")

# Test with a perfect cancer patient
feat_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
cancer_feat = {f: 0.0 for f in feat_names}
cancer_feat.update({
    "has_prior_chemo": 1.0, "has_metastatic_disease": 1.0,
    "tumor_condition_count": 0.3, "cancer_medication_count": 0.2,
    "diagnosis_overlap_score": 0.8, "lab_completeness": 0.5,
    "condition_count": 0.4, "medication_count": 0.3,
    "age_normalized": 0.55, "age_gap": 0.0,
})
X1 = np.array([[cancer_feat.get(f, 0.0) for f in feat_names]])
pred1 = model.predict(X1)
print(f"Perfect cancer patient score: {pred1[0]:.4f}")

# Test with a non-cancer patient  
no_cancer_feat = {f: 0.0 for f in feat_names}
no_cancer_feat.update({
    "lab_completeness": 0.5, "condition_count": 0.1,
    "medication_count": 0.1, "age_normalized": 0.45, "age_gap": 0.0,
})
X2 = np.array([[no_cancer_feat.get(f, 0.0) for f in feat_names]])
pred2 = model.predict(X2)
print(f"Non-cancer patient score: {pred2[0]:.4f}")
print(f"Score difference: {pred1[0] - pred2[0]:.4f}")

# Check evaluation metrics
try:
    with open("outputs/evaluation_metrics_cancer.json") as f:
        metrics = json.load(f)
    print(f"\nEvaluation metrics: {json.dumps(metrics, indent=2)}")
except FileNotFoundError:
    print("No cancer metrics file found")
