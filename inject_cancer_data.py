"""Inject realistic cancer conditions and medications into the Synthea CSV data.

Adds:
- Various cancer diagnoses to ~80-100 random patients
- Chemo/targeted therapy medications to cancer patients
- Metastatic disease conditions for some cancer patients
- Radiation therapy entries for some cancer patients

Uses a fixed seed for reproducibility.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(2026)
np.random.seed(2026)

DATA_DIR = "data/syntheaCSV-20260321T160919Z-3-001/syntheaCSV"

# Load existing data
patients = pd.read_csv(f"{DATA_DIR}/patients.csv")
conditions = pd.read_csv(f"{DATA_DIR}/conditions.csv")
medications = pd.read_csv(f"{DATA_DIR}/medications.csv")

# Find patients who already have cancer
cancer_kw = ['cancer', 'carcinoma', 'neoplasm', 'tumor', 'tumour', 'malignant', 'lymphoma', 'leukemia', 'melanoma', 'sarcoma']
existing_cancer_pids = set(
    conditions[conditions['DESCRIPTION'].str.lower().str.contains('|'.join(cancer_kw), na=False)]['PATIENT'].unique()
)
print(f"Existing cancer patients: {len(existing_cancer_pids)}")

# Select ~100 non-cancer patients to receive cancer conditions
non_cancer_pids = [pid for pid in patients['Id'].tolist() if pid not in existing_cancer_pids]
n_new_cancer = 100
selected_pids = random.sample(non_cancer_pids, min(n_new_cancer, len(non_cancer_pids)))
print(f"Adding cancer to {len(selected_pids)} new patients")

# Cancer condition templates (SNOMED-coded, realistic for Synthea)
CANCER_CONDITIONS = [
    (363406005, "Malignant tumor of colon (disorder)"),
    (254837009, "Malignant neoplasm of breast (disorder)"),
    (93761005, "Primary malignant neoplasm of colon (disorder)"),
    (126906006, "Neoplasm of prostate (disorder)"),
    (92546004, "Cancer of lung (disorder)"),
    (254632001, "Non-small cell lung cancer (disorder)"),
    (93655004, "Malignant neoplasm of pancreas (disorder)"),
    (363443007, "Malignant tumor of ovary (disorder)"),
    (109989006, "Squamous cell carcinoma of skin (disorder)"),
    (93143009, "Leukemia, disease (disorder)"),
    (109838007, "Overlapping malignant neoplasm of colon (disorder)"),
    (94260004, "Metastatic malignant neoplasm to colon (disorder)"),
    (254637007, "Non-small cell carcinoma of lung, TNM stage 1 (disorder)"),
    (271566001, "Suspected lung cancer (situation)"),
    (399068003, "Malignant tumor of urinary bladder (disorder)"),
    (363518003, "Malignant tumor of kidney (disorder)"),
    (188725004, "Hodgkin lymphoma (disorder)"),
    (118600007, "Non-Hodgkin lymphoma (disorder)"),
    (93143009, "Melanoma of skin (disorder)"),
    (94381002, "Metastatic malignant neoplasm to liver (disorder)"),
]

METASTATIC_CONDITIONS = [
    (94260004, "Metastatic malignant neoplasm to colon (disorder)"),
    (94381002, "Metastatic malignant neoplasm to liver (disorder)"),
    (94503003, "Metastatic malignant neoplasm to lung (disorder)"),
    (94222008, "Metastatic malignant neoplasm to bone (disorder)"),
    (94391008, "Metastatic malignant neoplasm to brain (disorder)"),
]

CHEMO_MEDICATIONS = [
    (387420009, "cyclophosphamide 50 MG Oral Tablet"),
    (387318005, "doxorubicin 20 MG Injection"),
    (386906001, "paclitaxel 100 MG Injection"),
    (387318005, "cisplatin 50 MG Injection"),
    (387195005, "carboplatin 150 MG Injection"),
    (387207008, "fluorouracil 500 MG Injection"),
    (387381009, "gemcitabine 200 MG Injection"),
    (386911004, "docetaxel 80 MG Injection"),
    (387512004, "capecitabine 500 MG Oral Tablet"),
    (386846001, "nivolumab 100 MG Injection"),
    (716298006, "pembrolizumab 100 MG Injection"),
    (387374002, "trastuzumab 440 MG Injection"),
    (387399001, "bevacizumab 400 MG Injection"),
    (386919002, "rituximab 500 MG Injection"),
    (387173000, "etoposide 50 MG Oral Capsule"),
    (716300001, "atezolizumab 840 MG Injection"),
]

RADIATION_CONDITION = (108290001, "Radiation therapy (procedure)")

# Get encounter mapping for selected patients
encounters = pd.read_csv(f"{DATA_DIR}/encounters.csv")

def get_random_encounter(pid):
    """Get a random encounter ID for a patient, or create a plausible one."""
    patient_encounters = encounters[encounters['PATIENT'] == pid]
    if len(patient_encounters) > 0:
        return patient_encounters.sample(1).iloc[0]['Id']
    return pid.replace('-', '')[:36]  # fallback

def random_date(start_year=2018, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    rand_days = random.randint(0, delta.days)
    return (start + timedelta(days=rand_days)).strftime("%Y-%m-%d")

# Build new condition rows
new_condition_rows = []
new_medication_rows = []

for pid in selected_pids:
    enc_id = get_random_encounter(pid)
    start_date = random_date(2018, 2024)
    
    # Primary cancer diagnosis (1-2 types)
    n_cancers = random.choices([1, 2], weights=[0.7, 0.3])[0]
    cancer_picks = random.sample(CANCER_CONDITIONS, n_cancers)
    for code, desc in cancer_picks:
        new_condition_rows.append({
            'START': start_date,
            'STOP': '',
            'PATIENT': pid,
            'ENCOUNTER': enc_id,
            'SYSTEM': 'http://snomed.info/sct',
            'CODE': code,
            'DESCRIPTION': desc,
        })
    
    # 30% chance of metastatic disease
    if random.random() < 0.30:
        meta_code, meta_desc = random.choice(METASTATIC_CONDITIONS)
        meta_date = random_date(int(start_date[:4]), 2025)
        new_condition_rows.append({
            'START': meta_date,
            'STOP': '',
            'PATIENT': pid,
            'ENCOUNTER': enc_id,
            'SYSTEM': 'http://snomed.info/sct',
            'CODE': meta_code,
            'DESCRIPTION': meta_desc,
        })
    
    # 20% chance of radiation therapy reference
    if random.random() < 0.20:
        new_condition_rows.append({
            'START': start_date,
            'STOP': '',
            'PATIENT': pid,
            'ENCOUNTER': enc_id,
            'SYSTEM': 'http://snomed.info/sct',
            'CODE': RADIATION_CONDITION[0],
            'DESCRIPTION': RADIATION_CONDITION[1],
        })
    
    # 50% chance of chemo medications (1-3)
    if random.random() < 0.50:
        n_meds = random.randint(1, 3)
        med_picks = random.sample(CHEMO_MEDICATIONS, min(n_meds, len(CHEMO_MEDICATIONS)))
        
        # Get a sample medication row format
        patient_meds = medications[medications['PATIENT'] == pid]
        for med_code, med_desc in med_picks:
            med_start = start_date
            med_stop = random_date(int(start_date[:4]) + 1, 2025)
            # Use the full medication CSV column format
            new_medication_rows.append({
                'START': med_start,
                'STOP': med_stop,
                'PATIENT': pid,
                'PAYER': '',
                'ENCOUNTER': enc_id,
                'CODE': med_code,
                'DESCRIPTION': med_desc,
                'BASE_COST': round(random.uniform(100, 5000), 2),
                'PAYER_COVERAGE': 0.0,
                'DISPENSES': random.randint(1, 12),
                'TOTALCOST': round(random.uniform(500, 50000), 2),
                'REASONCODE': '',
                'REASONDESCRIPTION': '',
            })

# Also add chemo meds to some EXISTING cancer patients who don't have them
existing_cancer_list = list(existing_cancer_pids)
random.shuffle(existing_cancer_list)
for pid in existing_cancer_list[:40]:
    enc_id = get_random_encounter(pid)
    start_date = random_date(2018, 2024)
    n_meds = random.randint(1, 2)
    med_picks = random.sample(CHEMO_MEDICATIONS, min(n_meds, len(CHEMO_MEDICATIONS)))
    for med_code, med_desc in med_picks:
        med_stop = random_date(int(start_date[:4]) + 1, 2025)
        new_medication_rows.append({
            'START': start_date,
            'STOP': med_stop,
            'PATIENT': pid,
            'PAYER': '',
            'ENCOUNTER': enc_id,
            'CODE': med_code,
            'DESCRIPTION': med_desc,
            'BASE_COST': round(random.uniform(100, 5000), 2),
            'PAYER_COVERAGE': 0.0,
            'DISPENSES': random.randint(1, 12),
            'TOTALCOST': round(random.uniform(500, 50000), 2),
            'REASONCODE': '',
            'REASONDESCRIPTION': '',
        })

print(f"New condition rows: {len(new_condition_rows)}")
print(f"New medication rows: {len(new_medication_rows)}")

# Append to CSVs
if new_condition_rows:
    new_conds_df = pd.DataFrame(new_condition_rows)
    # Ensure column order matches
    new_conds_df = new_conds_df[conditions.columns]
    updated_conditions = pd.concat([conditions, new_conds_df], ignore_index=True)
    updated_conditions.to_csv(f"{DATA_DIR}/conditions.csv", index=False)
    print(f"Updated conditions.csv: {len(conditions)} -> {len(updated_conditions)} rows")

if new_medication_rows:
    new_meds_df = pd.DataFrame(new_medication_rows)
    # Ensure column order matches
    new_meds_df = new_meds_df[medications.columns]
    updated_medications = pd.concat([medications, new_meds_df], ignore_index=True)
    updated_medications.to_csv(f"{DATA_DIR}/medications.csv", index=False)
    print(f"Updated medications.csv: {len(medications)} -> {len(updated_medications)} rows")

# Verify
print("\n--- Verification ---")
final_conds = pd.read_csv(f"{DATA_DIR}/conditions.csv")
final_cancer = final_conds[final_conds['DESCRIPTION'].str.lower().str.contains('|'.join(cancer_kw), na=False)]
print(f"Total cancer condition rows: {len(final_cancer)}")
print(f"Unique cancer patients: {final_cancer['PATIENT'].nunique()}")

final_meds = pd.read_csv(f"{DATA_DIR}/medications.csv")
chemo_kw = ['cyclophosphamide', 'doxorubicin', 'paclitaxel', 'cisplatin', 'carboplatin',
            'fluorouracil', 'gemcitabine', 'docetaxel', 'capecitabine', 'nivolumab',
            'pembrolizumab', 'trastuzumab', 'bevacizumab', 'rituximab', 'etoposide', 'atezolizumab']
final_chemo = final_meds[final_meds['DESCRIPTION'].str.lower().str.contains('|'.join(chemo_kw), na=False)]
print(f"Total chemo medication rows: {len(final_chemo)}")
print(f"Unique patients with chemo: {final_chemo['PATIENT'].nunique()}")
