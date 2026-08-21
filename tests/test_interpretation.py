import pytest
from gdsc.interpretation import ridge_bootstrap_stability, ridge_coefficients
from gdsc.models import build_ridge_model


def test_ridge_coefficients_rank_and_align():
    model = build_ridge_model().fit([[1, 0], [0, 1], [1, 1]], [2, 0, 2])
    result = ridge_coefficients(model, ["b", "a"])
    assert result.ABS_COEFFICIENT.is_monotonic_decreasing
    with pytest.raises(ValueError): ridge_coefficients(model, ["only"])


def test_bootstrap_stability_is_reproducible():
    args = (lambda: build_ridge_model(alpha=100), [[1,0],[0,1],[1,1],[2,1]], [1,0,1,2], ["g1","g2"])
    assert ridge_bootstrap_stability(*args, n_resamples=4, random_state=3).equals(ridge_bootstrap_stability(*args, n_resamples=4, random_state=3))
