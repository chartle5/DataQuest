# DataQuest Codebase Audit

**Date:** March 21, 2026

---

## What's Completed

### Project Architecture
- Clean 4-module structure (`trials/`, `patients/`, `matching/`, `pipelines/`) mapping to the spec's pipeline stages
- Pydantic-based schema for trial criteria and records (`src/trials/schema.py`)
- Config module with path management (`src/config.py`)
- Unit tests for API client, matching features, NER, and trial parser

### Phase 1: Data Collection
- ClinicalTrials.gov v2 API client with pagination (`src/trials/api.py`)
- JSON caching to avoid repeated API calls
- Synthea CSV data present for patients, conditions, medications, observations, encounters, and more
- Scoped to type 2 diabetes as the disease area

### Phase 2: Criteria Structuring
- Regex-based parser extracts age ranges, sex, HbA1c thresholds from free text (`src/trials/parser.py`)
- Inclusion/exclusion text splitting
- NER fallback using spaCy/scispacy for entity extraction (`src/trials/ner.py`)
- Graceful degradation to blank spaCy model if scispacy model is unavailable
- Detection of diabetes terms, renal failure terms, stroke terms, insulin pump terms
- Insulin allowance inference

### Phase 3: Feature Engineering
- Patient profile builder ingests 5 Synthea CSVs: patients, conditions, medications, observations, encounters (`src/patients/features.py`)
- Per-pair binary match features computed: `age_match`, `sex_match`, `t2d_present`, `hba1c_above_min`, `hba1c_below_max`, `renal_exclusion_hit`, `stroke_exclusion_hit`, `insulin_pump_conflict`, `missing_hba1c`, `missing_renal_lab` (`src/matching/features.py`)

### Phase 4: Modeling
- LGBMRanker (LambdaRank objective) with GradientBoostingRegressor fallback (`src/matching/ranker.py`)
- Rule-based labeling function for generating synthetic labels (`src/pipelines/build_pairs.py`)

### Pipeline Orchestration
- End-to-end `build_patient_trial_pairs()` function that loads patients, fetches trials, computes features, generates labels, and saves to CSV
- `rank_patients()` function that trains ranker and produces scored/ranked output

### Frontend
- Static HTML/CSS/JS prototype with condition selector and top-N input (`frontend/`)
- Dark-themed UI with form submission placeholder

---

## What Needs to Be Done

### Blocking: Environment Fix
- [ ] **Python version incompatibility** — spaCy 3.8 crashes on Python 3.14 due to pydantic v1 internals. Downgrade to Python 3.12 or wait for a compatible spaCy release.
- [ ] **Missing pandas** — `pandas` is in `requirements.txt` but not installed. Run `pip install pandas`.
- [ ] **Missing scikit-learn** — `scikit-learn` is in `requirements.txt` but not installed (needed by `ranker.py`).

### Bug Fixes
- [ ] **`TrialRecord.dict()` deprecated** — In `src/trials/api.py` line 68, replace `.dict()` with `.model_dump()` (Pydantic v2).
- [ ] **`datetime.utcnow()` deprecated** — In `src/patients/features.py` line 20, replace with `datetime.now(datetime.timezone.utc)`.
- [ ] **No train/test split** — `rank_patients()` in `src/pipelines/build_pairs.py` trains and predicts on the same data. Add a proper train/test split.
- [ ] **"Recent hospitalization" has no temporal filter** — `src/patients/features.py` line 67 marks any inpatient encounter as recent. Filter to encounters within a configurable time window (e.g., 6 months).
- [ ] **Stroke exclusion too strict** — `src/trials/parser.py` requires both a stroke term AND "recent" language. A trial saying just "Exclusion: stroke" won't trigger the flag. Make the "recent" requirement optional or add a separate `excludes_stroke` field.
- [ ] **API `fields` parameter uses v1-style names** — `src/trials/api.py` line 13 uses `NCTId,BriefTitle,...`. Verify against ClinicalTrials.gov v2 docs and update if needed.

### Phase 3: Feature Engineering (Incomplete)
- [ ] **Add aggregate match features** — The spec requires:
  - `inclusion_satisfied_count`
  - `exclusion_conflict_count`
  - `unknown_field_count`
  - Normalized age gap (how far patient age is from trial range)
  - Normalized lab gap (how far HbA1c is from threshold)
  - `diagnosis_overlap_score`
- [ ] **Improve condition matching** — Current substring matching (`keyword in c`) is fragile. Add SNOMED/ICD code mapping or fuzzy matching for Synthea condition descriptions.

### Phase 4: Modeling (Incomplete)
- [ ] **Add logistic regression baseline** — Spec requires this as the simplest model for comparison.
- [ ] **Add random forest baseline** — Spec requires this as a second baseline.
- [ ] **Add XGBoost as the main model** — Spec calls for XGBoost alongside or instead of LightGBM.
- [ ] **Support 3-class labeling** — Current `_rule_label()` returns only 0/1. Add a third class (2 = uncertain) for patients with missing data fields like `missing_hba1c` or `missing_renal_lab`.

### Phase 5: Evaluation (Not Started)
- [ ] **Implement evaluation metrics** — Precision, Recall, F1, ROC-AUC.
- [ ] **Implement precision@k** — For shortlist quality (e.g., "of the top 10 ranked patients, how many are actually eligible?").
- [ ] **Add cross-validation** — k-fold CV to assess model stability.
- [ ] **Model comparison report** — Compare logistic regression vs. random forest vs. XGBoost vs. LGBMRanker.

### Phase 6: Explainability (Not Started)
- [ ] **Feature importance** — Extract and display top contributing features from tree models.
- [ ] **Per-match reason codes** — For each patient-trial pair, output which criteria passed, failed, or are unknown (e.g., "age within range", "HbA1c threshold met", "renal status unknown").
- [ ] **Uncertainty reporting** — Flag patients where missing data makes the match ambiguous.

### Output Views (Not Started)
- [ ] **Trial coordinator view** — "Top N patients for Trial X" with match scores and reason codes.
- [ ] **Patient-centered view** — "Top N trials for Patient X" with match scores and reason codes.

### Backend Server (Not Started)
- [ ] **Build Streamlit app** — `streamlit` is in `requirements.txt` but no app file exists. Create a Streamlit dashboard that:
  - Lets user select a trial or patient
  - Displays ranked matches with scores
  - Shows reason codes for each match
  - Highlights uncertain/missing information

### Frontend Integration (Not Started)
- [ ] **Connect frontend to backend** — `frontend/script.js` currently shows "Backend coming soon." Wire it to the Streamlit or a Flask/FastAPI backend.
- [ ] **Display real results** — Show ranked patients/trials with scores and reason breakdowns.

### Temporal Logic (Not Started)
- [ ] **Time-aware criteria parsing** — Parse phrases like "within 6 months", "past 30 days", "recent" into structured time windows.
- [ ] **Time-aware patient features** — Filter conditions, medications, encounters by date relative to a reference point.

### Test Coverage Gaps
- [ ] **Tests for `patients/features.py`** — No tests for patient profile building or age calculation.
- [ ] **Tests for `pipelines/build_pairs.py`** — No tests for pair building, rule labeling, or ranking.
- [ ] **Tests for `matching/ranker.py`** — No tests for the ranker model.
- [ ] **Integration test** — End-to-end test that runs the full pipeline on a small sample.

---

## Summary

| Area | Status |
|---|---|
| Architecture | Done |
| Data collection (API + Synthea) | Done |
| Criteria parsing (regex + NER) | Done |
| Patient profiling | Done |
| Basic match features | Done |
| Aggregate match features | Not started |
| Rule-based labeling | Done (binary only, needs 3-class) |
| ML ranking model | 1 of 3+ models done |
| Train/test split | Not done |
| Evaluation metrics | Not started |
| Explainability / reason codes | Not started |
| Patient-centered view | Not started |
| Trial coordinator view | Not started |
| Streamlit backend | Not started |
| Frontend integration | Not started |
| Temporal logic | Not started |
| Environment runs cleanly | Blocked (Python 3.14 + spaCy) |

**Estimated completion: ~40-50% of the full spec.**
