from unittest.mock import patch

from src.trials.api import fetch_trials


def test_fetch_trials_parses_response():
    payload = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000001",
                        "briefTitle": "Diabetes Study",
                    },
                    "eligibilityModule": {"eligibilityCriteria": "Inclusion: adults"},
                    "conditionsModule": {"conditions": ["Diabetes"]},
                }
            }
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = payload
        mock_get.return_value.raise_for_status.return_value = None
        trials = fetch_trials("diabetes", limit=1)

    assert len(trials) == 1
    assert trials[0].trial_id == "NCT00000001"
