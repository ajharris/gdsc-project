"""Small, fixed-parameter baseline model constructors.

These helpers never preprocess data, split data, or evaluate on a test set.
Callers fit on training inputs and select approaches using validation only.
"""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV


def build_dummy_regressor() -> DummyRegressor:
    """Return the training-target mean baseline."""
    return DummyRegressor(strategy="mean")


def build_ridge_model(*, alpha: float = 1.0) -> Ridge:
    """Return a regularized genomic linear baseline; alpha is configurable."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return Ridge(alpha=alpha)


def build_elastic_net_model(*, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 10_000, random_state: int = 42) -> ElasticNet:
    """Return a fixed sparse-linear baseline without cross-validation."""
    if alpha < 0 or not 0 <= l1_ratio <= 1:
        raise ValueError("alpha must be non-negative and l1_ratio must be in [0, 1]")
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state)


def build_random_forest_model(*, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, max_features: float = 1.0, random_state: int = 42) -> RandomForestRegressor:
    """Return the single permitted nonlinear baseline candidate."""
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state, n_jobs=-1)


def tune_training_only(estimator, parameter_grid: dict, X_train, y_train, *, cv: int = 3):
    """Tune with training-only CV using negative RMSE; never receives validation/test."""
    search = GridSearchCV(estimator, parameter_grid, scoring="neg_root_mean_squared_error", cv=cv, refit=True, n_jobs=-1, return_train_score=False)
    search.fit(X_train, y_train)
    scores = -search.cv_results_["mean_test_score"]
    stds = search.cv_results_["std_test_score"]
    best = search.best_index_
    return search.best_estimator_, {"parameters": search.best_params_, "cv_rmse_mean": float(scores[best]), "cv_rmse_std": float(stds[best])}
