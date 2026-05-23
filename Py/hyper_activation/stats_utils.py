"""Helper routines for statistical readouts and colors."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.random import default_rng
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVC
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def make_orthogonal_permutation(labels: np.ndarray, rng=None) -> np.ndarray:
    """Permute labels while keeping counts balanced across classes."""
    rng = rng or default_rng()
    labels = np.asarray(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_labels = len(unique_labels)
    relative_counts = counts / len(labels)

    permuted = np.full(labels.shape, fill_value=-1, dtype=int)
    num_assigned = np.zeros(n_labels, dtype=int)

    def _build_base_vector(k_each: np.ndarray) -> np.ndarray:
        parts = [np.full(k, idx, dtype=int) for idx, k in enumerate(k_each) if k > 0]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    for idx, (label, lbl_size) in enumerate(zip(unique_labels, counts)):
        lbl_idxs = np.where(labels == label)[0]
        k_each = np.floor(lbl_size * relative_counts).astype(int)
        n_added = k_each.sum()
        base_vector = _build_base_vector(k_each)
        if n_added > 0 and base_vector.size:
            perm_vector = rng.permutation(base_vector)
            permuted[lbl_idxs[:n_added]] = perm_vector
        num_assigned += k_each

    k_each_label = counts - num_assigned
    base_vector = _build_base_vector(k_each_label)
    if base_vector.size:
        perm_vector = rng.permutation(base_vector)
        permuted[permuted < 0] = perm_vector

    return unique_labels[permuted]


def _model_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def _auc_with_strategy(model, X: np.ndarray, y: np.ndarray, use_loo: bool) -> float:
    if len(np.unique(y)) < 2:
        return np.nan

    if use_loo and len(y) > 1:
        loo = LeaveOneOut()
        scores = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in loo.split(X):
            mdl = clone(model)
            mdl.fit(X[train_idx], y[train_idx])
            scores[test_idx] = _model_scores(mdl, X[test_idx])
    else:
        mdl = clone(model)
        mdl.fit(X, y)
        scores = _model_scores(mdl, X)

    try:
        return roc_auc_score(y, scores)
    except ValueError:
        return np.nan


def calc_p_classification(
    X: np.ndarray,
    y: np.ndarray,
    model,
    *,
    num_permutations: int,
    use_loo: bool,
    rng=None,
) -> float:
    """Permutation-based p-value using ROC-AUC."""
    rng = rng or default_rng()
    baseline = _auc_with_strategy(model, X, y, use_loo=use_loo)
    if not np.isfinite(baseline):
        return np.nan

    perm_scores = []
    for _ in range(num_permutations):
        permuted_labels = make_orthogonal_permutation(y, rng=rng)
        perm_score = _auc_with_strategy(model, X, permuted_labels, use_loo=use_loo)
        if np.isfinite(perm_score):
            perm_scores.append(perm_score)

    if not perm_scores:
        return np.nan

    greater = sum(score >= baseline for score in perm_scores)
    p_value = (greater + 1) / (len(perm_scores) + 1)
    return p_value


def calc_pvalues(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    num_permutations: int,
    use_loo: bool,
    reference_value: str = "median",
    enable_classification: bool = True,
) -> Dict[str, Tuple[float, ...]]:
    df = pd.DataFrame(dict(x=x, y=y, z=z))

    pvalues: Dict[str, Tuple[float, ...] | float] = {}
    rsquared: Dict[str, Tuple[float, ...] | float] = {}
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Single predictor models
            model_x = smf.ols("z ~ x", df).fit()
            pvalues["pX"] = model_x.f_pvalue
            rsquared["r2X"] = model_x.rsquared
            
            model_y = smf.ols("z ~ y", df).fit()
            pvalues["pY"] = model_y.f_pvalue
            rsquared["r2Y"] = model_y.rsquared
            
            # Multi-predictor models (with adjusted r-squared)
            model_xy = smf.ols("z ~ x + y", df).fit()
            model_xy_int = smf.ols("z ~ x*y", df).fit()
            pvalues["pXY"] = (model_xy.f_pvalue, model_xy_int.f_pvalue)
            rsquared["r2XY"] = (model_xy.rsquared, model_xy_int.rsquared)
            rsquared["r2XY_adj"] = (model_xy.rsquared_adj, model_xy_int.rsquared_adj)
    except Exception:
        pvalues["pX"] = pvalues["pY"] = np.nan
        pvalues["pXY"] = (np.nan, np.nan)
        rsquared["r2X"] = rsquared["r2Y"] = np.nan
        rsquared["r2XY"] = (np.nan, np.nan)
        rsquared["r2XY_adj"] = (np.nan, np.nan)

    pvalues["pCLS"] = (np.nan, np.nan)  # Default: disabled
    rsquared["ROC"] = (np.nan, np.nan)  # Default: disabled

    # Classification p-values and ROC scores are computed only if enabled
    if enable_classification:
        X = np.vstack((x, y)).T
        if reference_value == "mean":
            zref = np.mean(z)
        else:
            zref = np.median(z)
        labels = (z > zref).astype(float)

        try:
            # Calculate ROC scores
            roc_logreg = _auc_with_strategy(LogisticRegression(max_iter=500), X, labels, use_loo=use_loo)
            roc_svc = _auc_with_strategy(LinearSVC(), X, labels, use_loo=use_loo)
            rsquared["ROC"] = (roc_logreg, roc_svc)
            
            # Calculate p-values
            p_cls = (
                calc_p_classification(
                    X,
                    labels,
                    LogisticRegression(max_iter=500),
                    num_permutations=num_permutations,
                    use_loo=use_loo,
                ),
                calc_p_classification(
                    X,
                    labels,
                    LinearSVC(),
                    num_permutations=num_permutations,
                    use_loo=use_loo,
                ),
            )
            pvalues["pCLS"] = p_cls
        except Exception:
            pass  # Keep default nan values
    
    # Combine pvalues and rsquared into a single dict
    combined = {**pvalues, **rsquared}
    return combined


def make_color(z: np.ndarray, margin: float = 1.3) -> np.ndarray:
    zc = z.copy().astype(float)
    if zc.ptp() == 0:
        return np.tile(np.array([[0.2, 0.2, 0.8]]), (len(zc), 1))
    zc -= zc.min()
    zc /= zc.max()
    zc = np.clip((zc - 0.5) * margin + 0.5, 0, 1)
    r = 1 - zc
    b = zc
    g = np.zeros_like(b)
    return np.vstack((r, g, b)).T


def calc_partial_correlations(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate partial correlation matrix between x, y, and z.
    
    Returns a 3x3 symmetric matrix where:
    - [0,1] is partial correlation between x and y controlling for z
    - [0,2] is partial correlation between x and z controlling for y
    - [1,2] is partial correlation between y and z controlling for x
    """
    # Use residuals from regression to compute partial correlations
    def partial_corr(a, b, control):
        """Partial correlation between a and b, controlling for control."""
        try:
            # Regress a on control, b on control
            from sklearn.linear_model import LinearRegression
            reg_a = LinearRegression().fit(control.reshape(-1, 1), a)
            reg_b = LinearRegression().fit(control.reshape(-1, 1), b)
            resid_a = a - reg_a.predict(control.reshape(-1, 1))
            resid_b = b - reg_b.predict(control.reshape(-1, 1))
            corr_result = pearsonr(resid_a, resid_b)
            return corr_result.correlation if np.isfinite(corr_result.correlation) else np.nan
        except Exception:
            return np.nan
    
    # Build correlation matrix
    corr_matrix = np.eye(3)
    
    # x-y partial (controlling for z)
    corr_matrix[0, 1] = corr_matrix[1, 0] = partial_corr(x, y, z)
    
    # x-z partial (controlling for y)
    corr_matrix[0, 2] = corr_matrix[2, 0] = partial_corr(x, z, y)
    
    # y-z partial (controlling for x)
    corr_matrix[1, 2] = corr_matrix[2, 1] = partial_corr(y, z, x)
    
    return corr_matrix


def calc_partial_correlations(x: np.ndarray, y: np.ndarray, z: np.ndarray, labels: Tuple[str, str, str] = ("X", "Y", "Z")) -> np.ndarray:
    """
    Calculate partial correlation matrix between x, y, and z.
    
    Partial correlation between two variables controlling for the third.
    
    Args:
        x, y, z: Arrays of values for the three axes
        labels: Labels for the axes (for display)
    
    Returns:
        3x3 symmetric matrix of partial correlations
    """
    def partial_corr(a: np.ndarray, b: np.ndarray, control: np.ndarray) -> float:
        """Partial correlation between a and b, controlling for control."""
        try:
            # Regress a on control, b on control to get residuals
            control_2d = control.reshape(-1, 1)
            reg_a = LinearRegression().fit(control_2d, a)
            reg_b = LinearRegression().fit(control_2d, b)
            resid_a = a - reg_a.predict(control_2d)
            resid_b = b - reg_b.predict(control_2d)
            corr_result = pearsonr(resid_a, resid_b)
            return corr_result.correlation if np.isfinite(corr_result.correlation) else np.nan
        except Exception:
            return np.nan
    
    # Build symmetric correlation matrix
    corr_matrix = np.eye(3)
    
    # Partial correlations (controlling for the third variable)
    # x-y partial (controlling for z)
    corr_matrix[0, 1] = corr_matrix[1, 0] = partial_corr(x, y, z)
    
    # x-z partial (controlling for y)
    corr_matrix[0, 2] = corr_matrix[2, 0] = partial_corr(x, z, y)
    
    # y-z partial (controlling for x)
    corr_matrix[1, 2] = corr_matrix[2, 1] = partial_corr(y, z, x)
    
    return corr_matrix


def format_stats_text(pvalues: Dict[str, Tuple[float, ...] | float]) -> str:
    lines = []
    for key, val in pvalues.items():
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            val_str = ", ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in val)
        else:
            v = float(val)
            val_str = f"{v:.3f}" if np.isfinite(v) else "nan"
        lines.append(f"{key}: {val_str}")
    return "\n".join(lines)

