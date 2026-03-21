import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import TrialRecord


@dataclass(frozen=True)
class RagDoc:
    doc_id: str
    domain: str
    text: str


def _iter_jsonl_files(data_dir: Path) -> Iterable[Path]:
    return data_dir.glob("*.jsonl")


def load_rag_documents(data_dir: Path) -> List[RagDoc]:
    docs: List[RagDoc] = []
    for path in _iter_jsonl_files(data_dir):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                title = payload.get("title") or ""
                content = payload.get("content") or ""
                tags = payload.get("tags") or []
                text = " ".join([title, content, " ".join(tags)]).strip()
                docs.append(
                    RagDoc(
                        doc_id=str(payload.get("doc_id") or ""),
                        domain=str(payload.get("domain") or ""),
                        text=text,
                    )
                )
    return docs


def build_rag_index(docs: List[RagDoc]) -> Tuple[TfidfVectorizer, "csr_matrix", List[str]]:
    texts = [doc.text for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    doc_ids = [doc.doc_id for doc in docs]
    return vectorizer, matrix, doc_ids


def compute_rag_features(text: str, vectorizer: TfidfVectorizer, matrix) -> Dict[str, float]:
    if not text:
        return {"rag_sim_max": 0.0, "rag_sim_mean": 0.0}
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, matrix).flatten()
    return {"rag_sim_max": float(sims.max()), "rag_sim_mean": float(sims.mean())}


def build_trial_rag_features(trials: List[TrialRecord], data_dir: Path) -> Dict[str, Dict[str, float]]:
    docs = load_rag_documents(data_dir)
    if not docs:
        return {trial.trial_id: {"rag_sim_max": 0.0, "rag_sim_mean": 0.0} for trial in trials}

    vectorizer, matrix, _ = build_rag_index(docs)
    trial_features: Dict[str, Dict[str, float]] = {}

    for trial in trials:
        text = " ".join(
            [
                trial.title or "",
                " ".join(trial.conditions or []),
                trial.eligibility_text or "",
            ]
        ).strip()
        trial_features[trial.trial_id] = compute_rag_features(text, vectorizer, matrix)

    return trial_features
