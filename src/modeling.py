"""Two-model T-learner (L2-regularized logistic regression) and treatment propensity model.

Uses `scikit-learn` when installed (`StandardScaler` + `LogisticRegression(C=1.0, solver='lbfgs')`),
with an exact `scipy.optimize.minimize` L-BFGS-B fallback solving the identical L2-penalized
binary cross-entropy objective when running in an offline VM without `scikit-learn`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.schema import ALLOWED_FEATURES, extract_feature_matrix

try:
    import sklearn  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    HAS_SKLEARN = True
except ImportError:
    sklearn = None  # type: ignore
    LogisticRegression = None  # type: ignore
    StandardScaler = None  # type: ignore
    HAS_SKLEARN = False


def get_ml_backend_info() -> str:
    """Return active ML backend name and version for the run record."""
    if HAS_SKLEARN:
        return f"scikit-learn {sklearn.__version__}"
    import scipy

    return (
        f"scipy {scipy.__version__} exact L2-logistic L-BFGS-B "
        "(offline fallback; install scikit-learn for native sklearn execution)"
    )


class _ExactStandardScaler:
    """Population-std z-score scaler matching sklearn.preprocessing.StandardScaler."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_ExactStandardScaler":
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = X_arr.mean(axis=0)
        scale = X_arr.std(axis=0, ddof=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit on training data before transform.")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class _ExactL2LogisticClassifier:
    """Exact L2-penalized binary logistic regression matching scikit-learn's L-BFGS objective:
       min_{w, b} sum_i [-y_i log p_i - (1 - y_i) log(1 - p_i)] + (1 / (2*C)) * ||w||_2^2
    """

    def __init__(self, C: float = 1.0, max_iter: int = 500) -> None:
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ExactL2LogisticClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        unique_vals = np.unique(y_arr)
        if len(unique_vals) < 2:
            raise ValueError("Training subset must contain both y=0 and y=1 outcomes.")

        n_samples, n_features = X_arr.shape
        inv_c = 1.0 / self.C

        # Initialize intercept at empirical log-odds for rapid, numerically stable convergence
        p_bar = float(np.clip(y_arr.mean(), 1e-6, 1.0 - 1e-6))
        theta0 = np.zeros(n_features + 1, dtype=np.float64)
        theta0[-1] = np.log(p_bar / (1.0 - p_bar))

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w = theta[:-1]
            b = theta[-1]
            logits = X_arr @ w + b
            # Numerically stable softplus loss: log(1 + exp(z)) - y*z
            nll = np.sum(np.logaddexp(0.0, logits) - y_arr * logits)
            reg = 0.5 * inv_c * float(np.dot(w, w))
            probs = expit(logits)
            diff = probs - y_arr
            grad_w = X_arr.T @ diff + inv_c * w
            grad_b = float(np.sum(diff))
            grad = np.empty_like(theta)
            grad[:-1] = grad_w
            grad[-1] = grad_b
            # Scale by n_samples strictly for well-conditioned optimizer tolerances
            return (nll + reg) / n_samples, grad / n_samples

        res = minimize(
            loss_and_grad,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )
        self.coef_ = res.x[:-1].reshape(1, -1)
        self.intercept_ = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Classifier must be fit before calling predict_proba.")
        logits = np.asarray(X, dtype=np.float64) @ self.coef_.ravel() + self.intercept_
        p1 = expit(logits)
        return np.column_stack([1.0 - p1, p1])


def binary_log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Compute binary cross-entropy (log-loss)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_pred, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Compute mean squared probability error (Brier score)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


@dataclass
class FittedAllocationModels:
    """Container holding frozen train-fitted preprocessing, T-learner arms, and e(X) model."""

    scaler: object
    treated_model: object
    control_model: object
    propensity_model: object
    pooled_train_e: float
    regularization_C: float
    feature_cols: tuple[str, ...]

    def predict_outcomes_and_uplift(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict p(visit | X, T=1), p(visit | X, T=0), uplift = p_treat - p_ctrl, and e(X)."""
        X_raw = extract_feature_matrix(df, self.feature_cols)
        X_scaled = self.scaler.transform(X_raw)
        p_treat = self.treated_model.predict_proba(X_scaled)[:, 1]
        p_ctrl = self.control_model.predict_proba(X_scaled)[:, 1]
        uplift = p_treat - p_ctrl
        e_hat_raw = self.propensity_model.predict_proba(X_scaled)[:, 1]
        return {
            "p_treat": p_treat,
            "p_ctrl": p_ctrl,
            "uplift_score": uplift,
            "e_hat": e_hat_raw,
        }


def _make_scaler() -> object:
    if HAS_SKLEARN:
        return StandardScaler()
    return _ExactStandardScaler()


def _make_logistic(C: float, seed: int) -> object:
    if HAS_SKLEARN:
        return LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500, random_state=seed)
    return _ExactL2LogisticClassifier(C=C, max_iter=500)


def fit_two_model_uplift_and_propensity(
    train_df: pd.DataFrame,
    C: float = 1.0,
    seed: int = 20261114,
    feature_cols: tuple[str, ...] = ALLOWED_FEATURES,
) -> FittedAllocationModels:
    """Fit scaler, treated logistic model, control logistic model, and e(X) on train split ONLY."""
    if "split" in train_df.columns and not (train_df["split"] == "train").all():
        raise ValueError("Model fitting received non-training rows! Split separation violated.")

    X_train_raw = extract_feature_matrix(train_df, feature_cols)
    t_train = train_df["treatment"].to_numpy(dtype=np.int8)
    y_train = train_df["visit"].to_numpy(dtype=np.int8)

    scaler = _make_scaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)

    treat_mask = t_train == 1
    ctrl_mask = t_train == 0
    if treat_mask.sum() == 0 or ctrl_mask.sum() == 0:
        raise ValueError("Training data must include both treated and control users.")

    treated_model = _make_logistic(C=C, seed=seed)
    treated_model.fit(X_train[treat_mask], y_train[treat_mask])

    control_model = _make_logistic(C=C, seed=seed + 1)
    control_model.fit(X_train[ctrl_mask], y_train[ctrl_mask])

    propensity_model = _make_logistic(C=C, seed=seed + 2)
    propensity_model.fit(X_train, t_train)

    pooled_train_e = float(np.mean(t_train))
    if not (0.0 < pooled_train_e < 1.0):
        raise ValueError(f"Degenerate training treatment rate: {pooled_train_e}")

    return FittedAllocationModels(
        scaler=scaler,
        treated_model=treated_model,
        control_model=control_model,
        propensity_model=propensity_model,
        pooled_train_e=pooled_train_e,
        regularization_C=float(C),
        feature_cols=tuple(feature_cols),
    )


def compute_validation_diagnostics(
    models: FittedAllocationModels,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute log-loss, Brier score, and calibration-in-the-large on validation split."""
    if "split" in val_df.columns and not (val_df["split"] == "val").all():
        raise ValueError("Validation diagnostics received non-validation rows.")

    preds = models.predict_outcomes_and_uplift(val_df)
    t_val = val_df["treatment"].to_numpy(dtype=np.int8)
    y_val = val_df["visit"].to_numpy(dtype=np.int8)

    rows = []
    for arm_name, mask, p_col in [
        ("treated_visit_model", t_val == 1, preds["p_treat"]),
        ("control_visit_model", t_val == 0, preds["p_ctrl"]),
    ]:
        y_sub = y_val[mask]
        p_sub = p_col[mask]
        base_rate = float(np.mean(y_sub))
        null_pred = np.full_like(p_sub, fill_value=np.clip(base_rate, 1e-12, 1.0 - 1e-12))
        rows.append(
            {
                "model": arm_name,
                "n_val_rows": int(mask.sum()),
                "observed_rate": base_rate,
                "mean_predicted_prob": float(np.mean(p_sub)),
                "calibration_gap": float(np.mean(p_sub) - base_rate),
                "log_loss": binary_log_loss(y_sub, p_sub),
                "null_intercept_log_loss": binary_log_loss(y_sub, null_pred),
                "brier_score": brier_score(y_sub, p_sub),
            }
        )

    # Propensity model validation diagnostic
    p_e = preds["e_hat"]
    t_rate = float(np.mean(t_val))
    null_e = np.full_like(p_e, fill_value=np.clip(t_rate, 1e-12, 1.0 - 1e-12))
    rows.append(
        {
            "model": "propensity_model_e_x",
            "n_val_rows": int(len(t_val)),
            "observed_rate": t_rate,
            "mean_predicted_prob": float(np.mean(p_e)),
            "calibration_gap": float(np.mean(p_e) - t_rate),
            "log_loss": binary_log_loss(t_val, p_e),
            "null_intercept_log_loss": binary_log_loss(t_val, null_e),
            "brier_score": brier_score(t_val, p_e),
        }
    )

    return pd.DataFrame(rows)
