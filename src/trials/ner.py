from typing import List, Optional
import spacy


DEFAULT_MODEL = "en_core_sci_sm"
_CACHED_MODEL: Optional[spacy.language.Language] = None


def load_ner_model(model_name: Optional[str] = None) -> spacy.language.Language:
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and model_name is None:
        return _CACHED_MODEL
    name = model_name or DEFAULT_MODEL
    try:
        model = spacy.load(name)
    except OSError:
        model = spacy.blank("en")
    if model_name is None:
        _CACHED_MODEL = model
    return model


def extract_entities(text: str, model_name: Optional[str] = None) -> List[str]:
    if not text:
        return []
    nlp = load_ner_model(model_name)
    if "ner" not in nlp.pipe_names:
        return []
    try:
        doc = nlp(text)
    except (ValueError, KeyError):
        return []
    return [ent.text for ent in doc.ents]
