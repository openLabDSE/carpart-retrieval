"""Checks for the prototype's ranking and "I don't know" rule (pipeline/type_identifier.py), with fake embeddings."""
import numpy as np
import pytest

from pipeline.type_identifier import TypeIdentifier, decide, rank_types

EMB = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])  # unit vectors
TYPES = np.array(["oil filter", "spark plug", "oil filter"])


def test_rank_uses_best_image_per_type():
    ranked = rank_types(np.array([1.0, 0.0]), EMB, TYPES)
    assert ranked[0] == ("oil filter", 1.0)  # best of its two photos
    assert [t for t, _ in ranked] == ["oil filter", "spark plug"]


def test_answers_above_threshold_and_refuses_at_or_below():
    ranked = [("oil filter", 0.97), ("spark plug", 0.5)]
    assert decide(ranked, 0.96, "balanced")["answered"] is True
    assert decide(ranked, 0.97, "balanced")["answered"] is False  # equal is refused, as in A6
    refused = decide(ranked, 0.99, "careful")
    assert refused["type"] is None and refused["top"][0]["type"] == "oil filter"  # top types still shown


def test_stricter_mode_refuses_more():
    ident = TypeIdentifier(EMB, TYPES, {"careful": 0.99, "balanced": 0.9}, embed_fn=lambda im: np.array([0.8, 0.6]))
    assert ident.identify(None, "balanced")["answered"] is True
    assert ident.identify(None, "careful")["answered"] is True  # 1.0 > 0.99: exact gallery match
    ident2 = TypeIdentifier(EMB, TYPES, {"careful": 0.99, "balanced": 0.9}, embed_fn=lambda im: np.array([0.6, 0.8]))
    assert ident2.identify(None, "balanced")["answered"] is True  # 0.96 vs 0.9
    assert ident2.identify(None, "careful")["answered"] is False  # 0.96 vs 0.99


def test_guess_mode_never_refuses():
    ranked = [("oil filter", 0.2)]
    assert decide(ranked, -1.0, "guess")["answered"] is True


def test_unknown_mode_is_rejected():
    ident = TypeIdentifier(EMB, TYPES, {"careful": 0.9, "balanced": 0.8}, embed_fn=lambda im: np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        ident.identify(None, "reckless")
