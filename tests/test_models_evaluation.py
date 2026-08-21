import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

from gdsc.evaluation import evaluate_locked_model, evaluate_regression, fit_and_evaluate_validation
from gdsc.models import build_dummy_regressor, build_elastic_net_model, build_random_forest_model, build_ridge_model, tune_training_only


def test_dummy_uses_training_target_mean_only():
    model = build_dummy_regressor()
    assert isinstance(model, DummyRegressor) and model.strategy == "mean"
    fitted, metrics = fit_and_evaluate_validation(model, [[1], [2]], [1., 3.], [[999]], [99.])
    assert fitted.predict([[0]]).tolist() == [2.]
    assert metrics.mae == pytest.approx(97.) and np.isnan(metrics.pearson)


def test_regularized_models_are_configurable_and_high_dimensional():
    X, y = np.arange(15.).reshape(3, 5), np.array([1., 2., 3.])
    ridge = build_ridge_model(alpha=2.)
    elastic = build_elastic_net_model(alpha=.1, l1_ratio=.2, random_state=7)
    assert isinstance(ridge, Ridge) and ridge.alpha == 2.
    assert isinstance(elastic, ElasticNet) and elastic.l1_ratio == .2
    assert ridge.fit(X, y).predict(X[:1]).shape == (1,)
    assert elastic.fit(X, y).predict(X[:1]).shape == (1,)
    with pytest.raises(ValueError): build_ridge_model(alpha=-1)


def test_random_forest_and_training_only_tuning_are_reproducible():
    X, y = np.arange(30.).reshape(6, 5), np.arange(6.)
    forest = build_random_forest_model(n_estimators=10, max_depth=2, random_state=7)
    assert isinstance(forest, RandomForestRegressor) and forest.random_state == 7
    assert np.allclose(forest.fit(X, y).predict(X), build_random_forest_model(n_estimators=10, max_depth=2, random_state=7).fit(X, y).predict(X))
    model, result = tune_training_only(build_ridge_model(), {"alpha": [.1, 1.]}, X, y, cv=3)
    assert isinstance(model, Ridge) and result["parameters"]["alpha"] in {.1, 1.}


def test_evaluation_known_and_degenerate_cases():
    positive = evaluate_regression([1, 2, 3], [1, 2, 3])
    negative = evaluate_regression([1, 2, 3], [3, 2, 1])
    assert positive.mae == 0 and positive.rmse == 0 and positive.pearson == pytest.approx(1)
    assert negative.pearson == pytest.approx(-1) and negative.spearman == pytest.approx(-1)
    assert np.isnan(evaluate_regression([1, 2], [2, 2]).pearson)
    with pytest.raises(ValueError): evaluate_regression([1], [np.nan])


def test_locked_evaluation_uses_training_fit_and_aligns_predictions():
    metadata = __import__("pandas").DataFrame({"COSMIC_ID": [9, 10]})
    _, metrics, table = evaluate_locked_model(build_dummy_regressor(), [[0], [1]], [1., 3.], [[8], [9]], [10., 20.], metadata)
    assert table.predicted_auc.tolist() == [2., 2.] and table.residual.tolist() == [8., 18.]
    assert metrics.mae == pytest.approx(13.)
