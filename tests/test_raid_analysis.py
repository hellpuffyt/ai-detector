import numpy as np
import pytest

from conftest import load_module

ra = load_module("raid/src/run_analysis.py", "raid_run_analysis")


def test_tpr_at_fpr_takes_last_point_within_budget():
    fpr = np.array([0.0, 0.005, 0.02, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 1.0])
    assert ra.get_tpr_at_fpr(fpr, tpr, 0.01) == pytest.approx(0.4)
    assert ra.get_tpr_at_fpr(fpr, tpr, 0.05) == pytest.approx(0.6)


def test_committed_predictions_reproduce_readme_numbers(repo):
    df = ra.load_data(str(repo / "raid/analysis/predictions_10000.csv"))
    res = ra.get_results_by_domain(df)
    tpr1 = {d: ra.get_tpr_at_fpr(f, t, 0.01) for d, (f, t, _) in res.items()}
    assert tpr1["reviews"] == pytest.approx(0.865, abs=0.001)
    assert tpr1["wiki"] == pytest.approx(0.352, abs=0.001)
    assert tpr1["reddit"] == pytest.approx(0.007, abs=0.001)
