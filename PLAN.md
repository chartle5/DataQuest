# Plan: Fix Hard Constraints, Evaluation Leakage & Condition-Agnostic Matching

## TL;DR

Five critical problems:
1. **Hard constraint bypass**: Patients outside age range get "High" confidence (77%) — age/sex must be hard fails.
2. **Evaluation leakage**: Recall=1.0 because ground truth labels use the same features as training — circular logic.
3. **T2D hardcoding**: Features hardcode `t2d_present` — cancer trials still check for T2D instead of the trial's actual target condition.
4. **No ranker modes**: Diabetes and cancer share the same scoring weights/features — need distinct ranker modes with condition-appropriate feature weights and cancer-specific features.
5. **Bad cancer trials**: ClinicalTrials.gov "cancer" query returns PTSD, Hepatitis C, skin photodamage trials — need a targeted query for actual oncology drug trials.

Fix by enforcing hard constraints, breaking feature-label circularity, replacing `t2d_present` with generic `key_condition_present`, implementing dual ranker modes (diabetes vs cancer), adding cancer-specific CSV features, and using better trial search queries.

---

## Phase 1: Enforce Hard Constraints (Scoring & Confidence)

### Step 1: Hard-fail in `_compute_confidence()` — `src/server.py` ~L157
- **Bug**: Only `exclusion_conflict_count > 0` caps confidence at 15%. Age/sex failures are completely ignored.
- **Fix**: If `age_match == 0.0` → confidence = 0%, status = "ineligible". Same for `sex_match == 0.0`.
- If `key_condition_present == 0.0` → cap at 15% (soft, since conditions may be under-coded in EHR).

### Step 2: Hard-fail in `_heuristic_score()` — `src/server.py` ~L186
- **Bug**: `age_match` weight +0.20 and `age_gap` penalty -0.25 net to only ~0.05-0.45 loss — insufficient to disqualify.
- **Fix**: If `age_match == 0.0` or `sex_match == 0.0`, force score to floor value (-1.0). These patients must rank below all eligible candidates.

### Step 3: Validate `_confidence_label()` — `src/server.py` ~L170
- Already maps <20 → "ineligible". Confirm 0% correctly produces "ineligible" after Step 1. No code change expected.

---

## Phase 2: Fix Rule Labels & Break Circular Logic

### Step 4: Add age/sex hard exclusions to `_rule_label()` — BOTH files
- **Bug**: `_rule_label()` ignores `age_match` and `sex_match`. A patient outside the trial age range can still score label=2 or 3 because `inclusion_satisfied_count >= 3` from other fields.
- **Fix**: Add at top of function (with existing exclusion checks): `if age_match == 0.0: return 0` and `if sex_match == 0.0: return 0`.
- **Files**: `src/pipelines/build_pairs.py` ~L79, `src/matching/train_models.py` ~L14.

### Step 5: Remove label-defining features from training feature set
- **Root cause of Recall=1.0**: These 7 features define `_rule_label()` AND are in the training X matrix:
  - `key_condition_present` (was `t2d_present`), `inclusion_satisfied_count`, `renal_exclusion_hit`, `stroke_exclusion_hit`, `insulin_pump_conflict`, `missing_hba1c`, `missing_renal_lab`
- The model trivially memorizes the deterministic rule → perfect recall.
- **Fix**: Define a `LABEL_FEATURES` constant. Exclude from `feature_cols` when building X. Also add `age_match` and `sex_match` (now used in rule). The model learns from indirect signals only: `age_gap`, `hba1c_gap`, `diagnosis_overlap_score`, `lab_completeness`, `rag_sim_max`, `rag_sim_mean`, `medication_count`, `condition_count`, `age_normalized`, `hba1c_value`, `insulin_on_med`, `recent_hospitalization`.
- **Files**: `src/pipelines/build_pairs.py` ~L55 in `rank_patients()`, `src/matching/train_models.py` ~L50 in `train_for_condition()`.

### Step 6: Verify metrics self-correct
- After Steps 4-5, re-run training. Expect realistic metrics: ROC-AUC ~0.65-0.85, Recall ~0.70-0.90.

---

## Phase 3: Condition-Agnostic Matching (Cancer vs Diabetes)

### Step 7: Replace `t2d_present` with `key_condition_present` — `src/matching/features.py`
- **Bug**: `t2d_present` hardcodes `_has_condition(patient, "type 2 diabetes")`. Cancer trials rank patients by T2D status — meaningless. The `ranked_cancer_top5.json` output checks `t2d_present: 1.0` for a PTSD trial.
- **Fix**:
  - Add `condition_keywords: List[str]` parameter to `build_match_features()`, derived from `TrialRecord.conditions`.
  - Replace `features["t2d_present"] = _has_condition(patient, "type 2 diabetes")` with:
    ```python
    features["key_condition_present"] = 1.0 if any(
        any(kw in c for c in patient.conditions)
        for kw in condition_keywords
    ) else 0.0
    ```
  - Default `condition_keywords` derivation: lowercase the trial's conditions list.

### Step 8: Propagate condition keywords through the pipeline
- `api_rank()` in `src/server.py`: Extract keywords from `trial.conditions` → pass to `build_match_features()`.
- `build_patient_trial_pairs()` in `src/pipelines/build_pairs.py`: Extract from each `trial.conditions`.
- `train_for_condition()` in `src/matching/train_models.py`: Same.
- `inclusion_satisfied_count` stays the same: `age_match + sex_match + key_condition_present + hba1c_above_min + hba1c_below_max`.

### Step 9: Update `_build_reasons()` to use trial condition name — `src/server.py`
- "Key condition (T2D) present" → "Key condition (Type 2 Diabetes) present" or "Key condition (Breast Cancer) present".
- Requires passing the trial condition name string into `_build_reasons()`.

---

## Phase 4: Improve Reason Strings

### Step 10: Make reasons medically intuitive — `src/server.py` ~L124
- Pass `PatientProfile` and `TrialCriteria` into `_build_reasons()`.
- Age: `"Patient age 88, trial requires 18-65 (23 years over limit)"` instead of `"gap 0.46"`
- HbA1c: `"HbA1c 7.2%, trial requires 7.0-10.0%"` instead of `"HbA1c above minimum threshold"`
- Labs: `"Lab data: 4/6 key labs available (missing: creatinine, triglycerides)"` instead of `"Good lab data completeness"`
- Trial-specific meds: Check for Metformin/Alogliptin if in trial title
- Call site: `api_rank()` loop (~L237) must pass patient profile and criteria objects.

---

## Phase 5: Reweight Heuristic Scoring

### Step 11: Adjust weights in `_heuristic_score()` — `src/server.py` ~L186
- `age_match`: 0.20 → 0.25 (confirms eligibility)
- `key_condition_present` (was `t2d_present`): 0.18 → 0.20
- `hba1c_above_min`: 0.08 → 0.12 (important for diabetes trials)
- `age_gap` penalty: -0.25 → -0.35 (stronger continuous penalty near boundaries)
- `renal_exclusion_hit`: -0.30 → -0.35
- Rename `t2d_present` → `key_condition_present` in weights dict.

---

## Relevant Files

| File | Changes |
|------|---------|
| `src/server.py` | Hard fails in `_compute_confidence()` and `_heuristic_score()`. Weight updates. Reason strings. Condition keyword propagation in `api_rank()`. |
| `src/matching/features.py` | `build_match_features()` — replace `t2d_present` → `key_condition_present`, accept `condition_keywords` param. |
| `src/pipelines/build_pairs.py` | `_rule_label()` — add age/sex hard exclusions, rename t2d check. `rank_patients()` — exclude label features from training X. `build_patient_trial_pairs()` — pass condition keywords. |
| `src/matching/train_models.py` | `_rule_label()` — same fixes. `train_for_condition()` — exclude label features, pass condition keywords. |
| `src/trials/schema.py` | Read-only reference (`TrialCriteria`, `TrialRecord`). |
| `src/patients/features.py` | Read-only reference (`PatientProfile`). |

---

## Verification

1. `pytest tests/` — update assertions for renamed `key_condition_present` feature.
2. `POST /api/rank {"condition": "type 2 diabetes", "top_n": 10}` — no patient with "Age outside range" gets "high" status.
3. `POST /api/rank {"condition": "cancer", "top_n": 5}` — candidates scored by cancer-relevant conditions, NOT T2D.
4. Re-run `build_pairs.py` → label=0 count increases (age/sex failures excluded).
5. Re-run training → Recall < 1.0, F1 < 0.95.
6. Reason strings show real ages in years, HbA1c in %, medication names.
7. Manual: patient outside age range → "ineligible", 0% confidence.
8. Manual: cancer ranking output does not mention "T2D".

---

## Decisions

- **Age/Sex = Hard Fails**: 0% confidence, "ineligible". Medically required.
- **Missing key condition = Soft cap (15%)**: EHR under-coding is common.
- **Shared features, swapped key condition**: Cancer and diabetes use same feature structure. `key_condition_present` checks the trial's actual conditions.
- **Feature exclusion over Golden Dataset**: Immediate fix for leakage. Golden dataset deferred.
- **Stale models**: Delete `outputs/models/` and retrain after all fixes.
- **`_rule_label()` DRY**: Both copies must be updated identically. Optional: extract to shared module.

---

## Phase 6: Dual Ranker Modes (Diabetes vs Cancer)

### Step 12: Define `RANKER_MODES` config — `src/matching/features.py`
- **Problem**: Both conditions share identical scoring weights. Cancer trials don't need HbA1c or insulin checks — they need tumor-relevant features.
- **Fix**: Create a `RANKER_MODES` dict keyed by condition category:
  ```python
  RANKER_MODES = {
      "diabetes": {
          "condition_keywords": ["type 2 diabetes", "type ii diabetes", "t2d", "diabetes mellitus"],
          "key_labs": ["a1c", "glucose", "creatinine", "cholesterol", "triglycerides", "hemoglobin"],
          "extra_features": [],  # standard features sufficient
      },
      "cancer": {
          "condition_keywords": ["cancer", "carcinoma", "neoplasm", "tumor", "tumour", "malignant", "oncology", "lymphoma", "leukemia", "melanoma", "sarcoma"],
          "key_labs": ["hemoglobin", "white blood cell", "platelet", "creatinine", "albumin", "calcium"],
          "extra_features": ["has_prior_chemo", "has_prior_radiation", "has_metastatic_disease", "tumor_condition_count", "cancer_medication_count"],
      },
  }
  ```

### Step 13: Add cancer-specific feature extraction — `src/matching/features.py`
- **New features** (only computed when mode="cancer"):
  - `has_prior_chemo`: 1.0 if any medication contains chemo-related keywords (cyclophosphamide, doxorubicin, paclitaxel, cisplatin, carboplatin, fluorouracil, methotrexate, etc.)
  - `has_prior_radiation`: 1.0 if any condition contains "radiation" or any medication contains "radiation therapy"
  - `has_metastatic_disease`: 1.0 if any condition contains "metastatic" or "metastasis" or "stage iv"
  - `tumor_condition_count`: count of cancer-related conditions in patient profile / 10 (normalized)
  - `cancer_medication_count`: count of chemo/targeted-therapy medications / 10 (normalized)
- `build_match_features()` accepts optional `mode` parameter (default "diabetes" for backward compat).

### Step 14: Mode-specific heuristic weights — `src/server.py`
- **Fix**: `_heuristic_score()` accepts a `mode` parameter.
- Diabetes weights stay as-is (with `t2d_present` → `key_condition_present` rename).
- Cancer weights:
  ```python
  CANCER_WEIGHTS = {
      "age_match": 0.20,
      "sex_match": 0.06,
      "key_condition_present": 0.25,  # higher — cancer match is critical
      "has_metastatic_disease": 0.10,
      "has_prior_chemo": 0.08,
      "cancer_medication_count": 0.06,
      "tumor_condition_count": 0.05,
      "diagnosis_overlap_score": 0.10,
      "rag_sim_max": 0.04,
  }
  ```
  Penalties: renal (-0.20), stroke (-0.15), age_gap (-0.30).

### Step 15: Mode-aware `api_rank()` routing — `src/server.py`
- Detect mode from condition string: if "cancer" in condition → mode="cancer", else mode="diabetes".
- Pass mode to `build_match_features()`, `_heuristic_score()`, and `_build_reasons()`.
- Load mode-specific model: `cancer_model.pkl` vs `type_2_diabetes_model.pkl`.

### Step 16: Mode-aware `_build_reasons()` — `src/server.py`
- Cancer mode: show cancer-specific reasons like "Patient has prior chemotherapy history", "Metastatic disease present", "3 cancer-related conditions found".
- Diabetes mode: keeps existing HbA1c, insulin, T2D reasons.

---

## Phase 7: Cancer-Specific CSV Output Parameters

### Step 17: Extend CSV columns for cancer mode — `src/pipelines/build_pairs.py`
- **Problem**: `patient_trial_pairs.csv` only has T2D-relevant columns. Cancer trials need cancer-specific columns.
- **Fix**: When `mode="cancer"`, CSV includes additional columns:
  - `has_prior_chemo`, `has_prior_radiation`, `has_metastatic_disease`
  - `tumor_condition_count`, `cancer_medication_count`
  - `cancer_lab_completeness` (fraction of cancer-relevant labs: hemoglobin, WBC, platelets, creatinine, albumin, calcium)
- These columns come naturally from `build_match_features(mode="cancer")`.

### Step 18: Condition-specific output files — `src/pipelines/build_pairs.py`
- Save as `patient_trial_pairs_diabetes.csv` and `patient_trial_pairs_cancer.csv` instead of one shared file.
- Each pipeline run specifies which mode to build pairs for.

---

## Phase 8: Better Cancer Trial Search

### Step 19: Improve cancer trial query — `src/trials/api.py`
- **Problem**: Searching "cancer" returns PTSD (NCT01865123), Hepatitis C (NCT02706223), skin photodamage trials — ClinicalTrials.gov full-text search is too broad.
- **Fix**: Use a more specific query for cancer:
  - Query: `"breast cancer" OR "lung cancer" OR "colorectal cancer" OR "prostate cancer"` (common solid tumors with well-structured eligibility criteria).
  - Add `query.cond` parameter (condition-specific search) instead of `query.term` for more precise results.
  - Add post-fetch filtering: trial `conditions` list must contain at least one cancer-related keyword.
- **Post-filter function** `_filter_cancer_trials(trials)`:
  ```python
  CANCER_KEYWORDS = {"cancer", "carcinoma", "neoplasm", "tumor", "tumour", "malignant", "lymphoma", "leukemia", "melanoma", "sarcoma", "oncology"}
  return [t for t in trials if any(kw in " ".join(t.conditions).lower() for kw in CANCER_KEYWORDS)]
  ```

### Step 20: Force-refresh cancer trial cache
- Delete `outputs/trials_cancer.json` and re-fetch with the improved query.
- Verify: all returned trials have cancer-relevant conditions (no PTSD, no Hepatitis C).

---

## Phase 9: Condition-Specific Trial Richness

### Step 21: Mode-aware `_trial_richness()` — `src/server.py`
- **Problem**: `_trial_richness()` scores trials by T2D-specific criteria (requires_t2d +2, hba1c_min +2). Cancer trials score poorly because they don't have T2D or HbA1c constraints.
- **Fix**: Cancer mode scores by: age range (+2), sex requirement (+1), cancer keywords in conditions (+3), exclusion criteria count (+1 each), entities count (+2). HbA1c and T2D are irrelevant.

---

## Updated Relevant Files

| File | Changes |
|------|---------|
| `src/server.py` | Hard fails in `_compute_confidence()` and `_heuristic_score()`. Dual-mode weight dicts. Mode-aware `api_rank()`, `_build_reasons()`, `_trial_richness()`. |
| `src/matching/features.py` | `RANKER_MODES` config. `build_match_features()` — `mode` param, `key_condition_present`, cancer features (`has_prior_chemo`, `has_metastatic_disease`, etc.). Cancer-specific lab completeness. |
| `src/pipelines/build_pairs.py` | `_rule_label()` — age/sex hard exclusions, generic `key_condition_present`. Mode-aware pair building. Separate CSV outputs per mode. |
| `src/matching/train_models.py` | `_rule_label()` fixes. Mode-aware training. Feature exclusion from label. |
| `src/trials/api.py` | Cancer-specific query using `query.cond`. Post-fetch cancer trial filtering. |
| `src/trials/schema.py` | Read-only reference. |
| `src/patients/features.py` | Read-only reference. |

---

## Updated Verification

1. `pytest tests/` — update assertions for renamed `key_condition_present` feature and new cancer features.
2. `POST /api/rank {"condition": "type 2 diabetes", "top_n": 10}` — no patient with "Age outside range" gets "high" status.
3. `POST /api/rank {"condition": "cancer", "top_n": 5}` — candidates scored by cancer features, NOT T2D.
4. Cancer output includes `has_prior_chemo`, `has_metastatic_disease`, `tumor_condition_count`.
5. Cancer trials are actual oncology studies (no PTSD or Hepatitis C).
6. Metrics files are separated: `evaluation_metrics_diabetes.json`, `evaluation_metrics_cancer.json`.
7. Manual: patient outside age range → "ineligible", 0% confidence, regardless of mode.
8. Manual: cancer ranking output shows cancer-specific reasons.
9. Heuristic weights differ between modes.

---

## Updated Decisions

- **Age/Sex = Hard Fails**: 0% confidence, "ineligible". Medically required.
- **Missing key condition = Soft cap (15%)**: EHR under-coding is common.
- **Shared features, swapped key condition**: Cancer and diabetes use same feature structure. `key_condition_present` checks the trial's actual conditions.
- **Feature exclusion over Golden Dataset**: Immediate fix for leakage. Golden dataset deferred.
- **Stale models**: Delete `outputs/models/` and retrain after all fixes.
- **`_rule_label()` DRY**: Both copies must be updated identically. Optional: extract to shared module.
- **Dual ranker modes**: Diabetes and cancer have distinct weight profiles, feature sets, and trial quality checks.
- **Cancer query uses `query.cond`**: More precise than `query.term` for condition-specific trials.
- **Post-filtering**: Even with better queries, filter out trials without cancer keywords in their conditions list.

---

## Further Considerations

1. **Cancer-specific features (future)**: ECOG performance status, tumor stage, prior therapies from the oncology RAG corpus. Add when oncology trial criteria parsing is richer.
2. **Golden Dataset**: 50-100 expert-labeled patient-trial pairs for held-out validation.
3. **Additional cancer subtypes**: Separate modes for breast, lung, colorectal if trial criteria differ significantly.
4. **Mode auto-detection**: Infer mode from the trial's conditions list rather than requiring explicit mode in API call.
