"""Validation-only regression evaluation helpers."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    pearson: float
    spearman: float
    r2: float


def _correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(function(y_true, y_pred).statistic)


def evaluate_regression(y_true, y_pred) -> RegressionMetrics:
    """Return standard metrics; undefined correlations are explicitly NaN."""
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must be equally sized one-dimensional arrays")
    if len(actual) == 0 or not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("metrics require non-empty finite arrays")
    return RegressionMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(mean_squared_error(actual, predicted) ** .5),
        pearson=_correlation(pearsonr, actual, predicted),
        spearman=_correlation(spearmanr, actual, predicted),
        r2=float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    )


def fit_and_evaluate_validation(model, X_train, y_train, X_val, y_val):
    """Fit only supplied training inputs and return validation metrics."""
    fitted = model.fit(X_train, y_train)
    return fitted, evaluate_regression(y_val, fitted.predict(X_val))


def evaluate_locked_model(model, X_train, y_train, X_test, y_test, metadata_test):
    """Fit an already locked configuration on training data and evaluate once on test.

    This helper deliberately accepts no tuning parameters and does not touch
    validation data. Metadata is carried solely to make predictions auditable.
    """
    fitted = model.fit(X_train, y_train)
    prediction = fitted.predict(X_test)
    if len(prediction) != len(metadata_test):
        raise ValueError("Test metadata and predictions must have equal length")
    import pandas as pd
    table = metadata_test.copy()
    table["observed_auc"] = np.asarray(y_test)
    table["predicted_auc"] = prediction
    table["residual"] = table["observed_auc"] - table["predicted_auc"]
    return fitted, evaluate_regression(y_test, prediction), table
