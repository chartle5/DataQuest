import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import requests

from .schema import TrialRecord
from .parser import parse_eligibility_text


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Keywords that indicate a trial is actually about cancer
CANCER_FILTER_KEYWORDS: Set[str] = {
    "cancer", "carcinoma", "neoplasm", "tumor", "tumour", "malignant",
    "lymphoma", "leukemia", "leukaemia", "melanoma", "sarcoma", "oncology",
    "myeloma", "glioma", "glioblastoma", "adenocarcinoma",
}

# More targeted cancer queries to avoid generic hits
CANCER_CONDITION_QUERY = (
    "breast cancer OR lung cancer OR colorectal cancer OR prostate cancer "
    "OR pancreatic cancer OR ovarian cancer OR lymphoma OR leukemia OR melanoma"
)


def _build_params(query: str, page_token: Optional[str], page_size: int,
                  use_condition_field: bool = False) -> Dict[str, str]:
    params: Dict[str, str] = {
        "pageSize": str(page_size),
        "fields": "NCTId,BriefTitle,Condition,EligibilityCriteria",
    }
    if use_condition_field:
        params["query.cond"] = query
    else:
        params["query.term"] = query
    if page_token:
        params["pageToken"] = page_token
    return params


def _is_cancer_trial(trial: TrialRecord) -> bool:
    """Check if a trial's conditions actually relate to cancer."""
    cond_text = " ".join(trial.conditions).lower()
    title_text = trial.title.lower()
    combined = cond_text + " " + title_text
    return any(kw in combined for kw in CANCER_FILTER_KEYWORDS)


def fetch_trials(
    query: str,
    limit: int = 50,
    page_size: int = 50,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
    mode: str = "diabetes",
) -> List[TrialRecord]:
    if cache_path and cache_path.exists() and not force_refresh:
        return load_trials_json(str(cache_path))

    # Cancer mode: use condition-specific search with targeted query
    use_cond_field = (mode == "cancer")
    effective_query = CANCER_CONDITION_QUERY if mode == "cancer" else query

    trials: List[TrialRecord] = []
    next_token: Optional[str] = None
    # Fetch more than needed since post-filtering may drop irrelevant trials
    fetch_limit = limit * 3 if mode == "cancer" else limit

    while len(trials) < fetch_limit:
        params = _build_params(effective_query, next_token, page_size,
                               use_condition_field=use_cond_field)
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        studies = payload.get("studies", [])
        next_token = payload.get("nextPageToken")

        for study in studies:
            if len(trials) >= fetch_limit:
                break
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            eligibility = proto.get("eligibilityModule", {})
            conditions = proto.get("conditionsModule", {})

            trial_id = ident.get("nctId") or ""
            title = ident.get("briefTitle") or ""
            eligibility_text = eligibility.get("eligibilityCriteria") or ""
            cond_list = conditions.get("conditions", []) or []

            criteria = parse_eligibility_text(eligibility_text)
            record = TrialRecord(
                trial_id=trial_id,
                title=title,
                conditions=cond_list,
                eligibility_text=eligibility_text,
                criteria=criteria,
            )
            trials.append(record)

        if not next_token:
            break

    # Post-filter: for cancer mode, drop trials without cancer-related conditions
    if mode == "cancer":
        trials = [t for t in trials if _is_cancer_trial(t)]

    # Trim to requested limit
    trials = trials[:limit]

    if cache_path:
        save_trials_json(trials, str(cache_path))
    return trials


def save_trials_json(trials: Iterable[TrialRecord], path: str) -> None:
    data = [trial.model_dump() for trial in trials]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_trials_json(path: str) -> List[TrialRecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TrialRecord(**item) for item in data]
