import pytest
from gdsc.interpretation import ridge_bootstrap_stability, ridge_coefficients, top_feature_correlations
from gdsc.models import build_ridge_model


def test_ridge_coefficients_rank_and_align():
    model = build_ridge_model().fit([[1, 0], [0, 1], [1, 1]], [2, 0, 2])
    result = ridge_coefficients(model, ["b", "a"])
    assert result.ABS_COEFFICIENT.is_monotonic_decreasing
    with pytest.raises(ValueError): ridge_coefficients(model, ["only"])


def test_bootstrap_stability_is_reproducible():
    args = (lambda: build_ridge_model(alpha=100), [[1,0],[0,1],[1,1],[2,1]], [1,0,1,2], ["g1","g2"])
    assert ridge_bootstrap_stability(*args, n_resamples=4, random_state=3).equals(ridge_bootstrap_stability(*args, n_resamples=4, random_state=3))


def test_top_feature_correlations_exclude_self_and_rank_deterministically():
    import pandas as pd
    result = top_feature_correlations(pd.DataFrame({"a": [1,2,3], "b": [2,4,6], "c": [3,2,1]}), ["a", "b", "c"])
    assert len(result) == 3 and result.iloc[0].ABS_CORRELATION == pytest.approx(1)
    assert set(result.iloc[0][["GENE_A", "GENE_B"]]) == {"a", "b"}
