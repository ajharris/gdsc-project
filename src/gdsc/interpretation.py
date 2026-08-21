"""Post-lock feature interpretation; never tunes or changes the feature set."""
import numpy as np
import pandas as pd


def ridge_coefficients(model, feature_names) -> pd.DataFrame:
    """Rank fitted Ridge coefficients by absolute magnitude, preserving sign."""
    names, coefficients = list(feature_names), np.asarray(model.coef_)
    if len(names) != len(coefficients):
        raise ValueError("feature_names must align exactly with model coefficients")
    table = pd.DataFrame({"GENE_SYMBOL": names, "COEFFICIENT": coefficients})
    table["ABS_COEFFICIENT"] = table.COEFFICIENT.abs()
    table["SIGN"] = np.sign(table.COEFFICIENT).astype(int)
    return table.sort_values(["ABS_COEFFICIENT", "GENE_SYMBOL"], ascending=[False, True]).reset_index(drop=True)


def ridge_bootstrap_stability(model_factory, X_train, y_train, feature_names, *, n_resamples=100, random_state=42) -> pd.DataFrame:
    """Refit the already locked factory on training-only bootstrap samples."""
    X, y, rng = np.asarray(X_train), np.asarray(y_train), np.random.default_rng(random_state)
    values = np.asarray([model_factory().fit(X[idx], y[idx]).coef_ for idx in (rng.integers(0, len(y), len(y)) for _ in range(n_resamples))])
    return pd.DataFrame({"GENE_SYMBOL": list(feature_names), "COEFFICIENT_MEAN": values.mean(0), "COEFFICIENT_MEDIAN": np.median(values, 0), "COEFFICIENT_STD": values.std(0), "POSITIVE_FRACTION": (values > 0).mean(0), "NEGATIVE_FRACTION": (values < 0).mean(0)})
