import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import requests

from .schema import TrialRecord
from .parser import parse_eligibility_text


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def _build_params(query: str, page_token: Optional[str], page_size: int) -> Dict[str, str]:
    params: Dict[str, str] = {
        "query.term": query,
        "pageSize": str(page_size),
        "fields": "NCTId,BriefTitle,Condition,EligibilityCriteria",
    }
    if page_token:
        params["pageToken"] = page_token
    return params


def fetch_trials(
    query: str,
    limit: int = 50,
    page_size: int = 50,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> List[TrialRecord]:
    if cache_path and cache_path.exists() and not force_refresh:
        return load_trials_json(str(cache_path))

    trials: List[TrialRecord] = []
    next_token: Optional[str] = None

    while len(trials) < limit:
        params = _build_params(query, next_token, page_size)
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        studies = payload.get("studies", [])
        next_token = payload.get("nextPageToken")

        for study in studies:
            if len(trials) >= limit:
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
