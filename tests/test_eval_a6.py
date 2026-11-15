"""Checks for the A6 threshold rule and rates (scripts/eval_a6.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_a6 import rate, risk_coverage, threshold_for_far  # noqa: E402


def test_threshold_keeps_far_at_or_below_target():
    s = np.linspace(0, 1, 200)  # 200 unknown scores
    thr = threshold_for_far(s, 0.05)
    assert (s > thr).mean() <= 0.05
    assert (s > thr - 0.01).mean() > 0.05  # and it is the lowest such threshold (within one step)


def test_ties_are_refused_so_far_holds():
    s = np.array([0.5] * 100)
    thr = threshold_for_far(s, 0.05)
    assert (s > thr).mean() == 0.0


def test_rate_answered_and_refused_sum_to_one():
    s = np.array([0.1, 0.4, 0.6, 0.9])
    assert rate(s, 0.5, True)["value"] + rate(s, 0.5, False)["value"] == 1.0
    assert rate(np.array([]), 0.5, True)["n"] == 0


def test_risk_falls_as_coverage_falls_for_a_good_score():
    score = np.linspace(0, 1, 100)
    correct = (score > 0.5).astype(float)  # confident answers are right
    rc = risk_coverage(score, correct)
    assert rc[0]["coverage"] > rc[-1]["coverage"]
    assert rc[-1]["risk"] <= rc[0]["risk"]
