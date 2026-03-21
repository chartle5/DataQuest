from src.trials.ner import extract_entities


def test_extract_entities_empty():
    assert extract_entities("") == []
