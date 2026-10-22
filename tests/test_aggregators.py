import math

import pytest

from conftest import load_module

import torch
agg = load_module("raid/src/aggregators.py", "raid_aggregators")


def naive_all_or_nothing(p):
    p_true = torch.prod(p)
    p_false = torch.prod(1 - p)
    return (p_true / (p_true + p_false)).item()


def test_all_or_nothing_matches_closed_form_on_small_inputs():
    p = torch.tensor([0.2, 0.7, 0.9])
    assert agg.all_or_nothing(p) == pytest.approx(naive_all_or_nothing(p), rel=1e-5)


def test_all_or_nothing_long_text_does_not_underflow_to_nan():
    # Regression: 60 paragraphs (30 at 0.001, 30 at 0.999) made both products
    # underflow to 0 in float32, giving 0/0 = NaN.
    p = torch.tensor([0.001] * 30 + [0.999] * 30)
    assert math.isnan(naive_all_or_nothing(p))
    # float32 0.999 and 0.001 are not exact complements, so allow a small tolerance.
    assert agg.all_or_nothing(p) == pytest.approx(0.5, abs=1e-3)


def test_all_or_nothing_handles_exact_zero_and_one():
    assert not math.isnan(agg.all_or_nothing(torch.tensor([0.0, 1.0])))


def test_simple_aggregators():
    p = torch.tensor([0.1, 0.5, 0.9])
    assert agg.take_max(p) == pytest.approx(0.9)
    assert agg.take_min(p) == pytest.approx(0.1)
    assert agg.take_mean(p) == pytest.approx(0.5)
