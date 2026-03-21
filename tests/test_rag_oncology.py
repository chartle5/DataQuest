import json
from pathlib import Path

from src.trials.rag import build_rag_index, build_trial_rag_features, compute_rag_features, load_rag_documents
from src.trials.schema import TrialCriteria, TrialRecord


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_trial_rag_features_oncology_similarity(tmp_path: Path) -> None:
    docs_path = tmp_path / "oncology.jsonl"
    _write_jsonl(
        docs_path,
        [
            {
                "doc_id": "onc-1",
                "domain": "oncology",
                "title": "Breast cancer trial",
                "content": "Study on metastatic breast cancer treatments",
                "tags": ["oncology", "breast cancer"],
            }
        ],
    )

    trial = TrialRecord(
        trial_id="NCT-ONC-001",
        title="Breast cancer trial",
        conditions=["Breast Cancer"],
        eligibility_text="Eligible adults with metastatic breast cancer",
        criteria=TrialCriteria(),
    )

    features = build_trial_rag_features([trial], tmp_path)
    assert features[trial.trial_id]["rag_sim_max"] > 0.2
    assert features[trial.trial_id]["rag_sim_mean"] > 0.0


def test_build_trial_rag_features_empty_docs(tmp_path: Path) -> None:
    trial = TrialRecord(
        trial_id="NCT-ONC-002",
        title="Lung cancer trial",
        conditions=["Lung Cancer"],
        eligibility_text="Eligibility criteria",
        criteria=TrialCriteria(),
    )

    features = build_trial_rag_features([trial], tmp_path)
    assert features[trial.trial_id] == {"rag_sim_max": 0.0, "rag_sim_mean": 0.0}


def test_compute_rag_features_empty_text(tmp_path: Path) -> None:
    docs_path = tmp_path / "oncology.jsonl"
    _write_jsonl(
        docs_path,
        [
            {
                "doc_id": "onc-2",
                "domain": "oncology",
                "title": "Ovarian cancer",
                "content": "Ovarian cancer immunotherapy study",
                "tags": ["oncology"],
            }
        ],
    )

    docs = load_rag_documents(tmp_path)
    vectorizer, matrix, _ = build_rag_index(docs)
    features = compute_rag_features("", vectorizer, matrix)

    assert features == {"rag_sim_max": 0.0, "rag_sim_mean": 0.0}


def test_load_rag_documents_multiple_files(tmp_path: Path) -> None:
    first = tmp_path / "oncology.jsonl"
    second = tmp_path / "oncology_more.jsonl"

    _write_jsonl(
        first,
        [
            {
                "doc_id": "onc-3",
                "domain": "oncology",
                "title": "Pancreatic cancer",
                "content": "Pancreatic cancer treatment",
                "tags": [],
            }
        ],
    )
    _write_jsonl(
        second,
        [
            {
                "doc_id": "onc-4",
                "domain": "oncology",
                "title": "Melanoma",
                "content": "Melanoma immune checkpoint trial",
                "tags": ["immunotherapy"],
            }
        ],
    )

    docs = load_rag_documents(tmp_path)
    doc_ids = {doc.doc_id for doc in docs}
    assert doc_ids == {"onc-3", "onc-4"}
