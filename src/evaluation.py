"""Horvitz-Thompson / IPW policy evaluation, sensitivity checks, and paired bootstrap."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from src.policies import (
    compute_budget_quota,
    select_top_k_deterministic,
    uniform_policy_probabilities,
)


def compute_ipw_pseudo_outcome(
    treatment: np.ndarray,
    visit: np.ndarray,
    propensity: np.ndarray | float,
) -> np.ndarray:
    """Compute Horvitz-Thompson pseudo-outcome:
       z_i = T_i * Y_i / e_i - (1 - T_i) * Y_i / (1 - e_i)
    """
    t = np.asarray(treatment, dtype=np.float64)
    y = np.asarray(visit, dtype=np.float64)
    if len(t) != len(y):
        raise ValueError("treatment and visit arrays must have identical length.")

    e = np.broadcast_to(np.asarray(propensity, dtype=np.float64), t.shape)
    if np.any((e <= 0.0) | (e >= 1.0) | ~np.isfinite(e)):
        raise ValueError("Propensity e_i must strictly lie in (0, 1) with no NaNs.")

    return (t * y) / e - ((1.0 - t) * y) / (1.0 - e)


def compute_policy_value_per_1k(
    selection_prob: np.ndarray,
    z_scores: np.ndarray,
) -> float:
    """Compute V(pi) = (1,000 / N) * sum_i pi_i * z_i across the FULL eligible cohort N."""
    pi = np.asarray(selection_prob, dtype=np.float64)
    z = np.asarray(z_scores, dtype=np.float64)
    if len(pi) != len(z):
        raise ValueError("selection_prob and z_scores must have identical length.")
    n = len(pi)
    if n == 0:
        raise ValueError("Cannot evaluate policy value on empty cohort (N=0).")
    return float((1000.0 / float(n)) * np.sum(pi * z))


def compute_overall_itt_visit_diff_pp(
    treatment: np.ndarray,
    visit: np.ndarray,
) -> dict[str, float]:
    """Compute raw treated-minus-control visit-rate difference in percentage points."""
    t = np.asarray(treatment, dtype=np.int8)
    y = np.asarray(visit, dtype=np.float64)
    t1 = t == 1
    t0 = t == 0
    if t1.sum() == 0 or t0.sum() == 0:
        raise ValueError("Held-out cohort must contain both treated and control rows.")
    rate_t1 = float(np.mean(y[t1]))
    rate_t0 = float(np.mean(y[t0]))
    return {
        "visit_rate_treated": rate_t1,
        "visit_rate_control": rate_t0,
        "itt_visit_diff_pp": 100.0 * (rate_t1 - rate_t0),
    }


def evaluate_three_policies(
    treatment: np.ndarray,
    visit: np.ndarray,
    p_treat: np.ndarray,
    uplift_score: np.ndarray,
    tie_breaker_ids: np.ndarray,
    propensity: np.ndarray | float,
    budget_share: float = 0.20,
    quota_override: Optional[int] = None,
) -> dict:
    """Evaluate Uniform, Response, and Uplift policies at identical quota K."""
    n = len(treatment)
    quota_k = (
        compute_budget_quota(n, budget_share=budget_share)
        if quota_override is None
        else int(quota_override)
    )

    pi_uniform = uniform_policy_probabilities(n, quota_k)
    pi_response = select_top_k_deterministic(p_treat, tie_breaker_ids, quota_k)
    pi_uplift = select_top_k_deterministic(uplift_score, tie_breaker_ids, quota_k)

    z = compute_ipw_pseudo_outcome(treatment, visit, propensity)

    v_uniform = compute_policy_value_per_1k(pi_uniform, z)
    v_response = compute_policy_value_per_1k(pi_response, z)
    v_uplift = compute_policy_value_per_1k(pi_uplift, z)

    return {
        "n_users": int(n),
        "quota_k": int(quota_k),
        "budget_share_realized": float(quota_k / n) if n > 0 else 0.0,
        "pi_uniform": pi_uniform,
        "pi_response": pi_response,
        "pi_uplift": pi_uplift,
        "z": z,
        "values_per_1k": {
            "uniform": v_uniform,
            "response": v_response,
            "uplift": v_uplift,
        },
        "differences_per_1k": {
            "uplift_vs_response": v_uplift - v_response,
            "uplift_vs_uniform": v_uplift - v_uniform,
            "response_vs_uniform": v_response - v_uniform,
        },
    }


def paired_bootstrap_evaluation(
    treatment: np.ndarray,
    visit: np.ndarray,
    p_treat: np.ndarray,
    uplift_score: np.ndarray,
    tie_breaker_ids: np.ndarray,
    propensity: np.ndarray | float,
    budget_share: float = 0.20,
    n_replicates: int = 1000,
    seed: int = 20261115,
    quota_override: Optional[int] = None,
) -> dict:
    """Run paired bootstrap on held-out users keeping fitted models fixed.

    In each replicate:
      - Resample N held-out users with replacement using ONE shared index draw for all policies.
      - Recompute K = floor(budget_share * N_draw) (unless quota_override is explicitly set).
      - Re-rank and re-select top-K users inside the resampled cohort for Response and Uplift.
      - Compute V(uniform), V(response), V(uplift), and all pairwise policy differences.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")

    t_arr = np.asarray(treatment, dtype=np.float64)
    y_arr = np.asarray(visit, dtype=np.float64)
    p1_arr = np.asarray(p_treat, dtype=np.float64)
    up_arr = np.asarray(uplift_score, dtype=np.float64)
    id_arr = np.asarray(tie_breaker_ids, dtype=np.int64)
    n = len(t_arr)
    e_arr = np.broadcast_to(np.asarray(propensity, dtype=np.float64), (n,)).copy()
    z_full = compute_ipw_pseudo_outcome(t_arr, y_arr, e_arr)

    rng = np.random.default_rng(seed)

    boot_values = {
        "uniform": np.empty(n_replicates, dtype=np.float64),
        "response": np.empty(n_replicates, dtype=np.float64),
        "uplift": np.empty(n_replicates, dtype=np.float64),
    }
    boot_diffs = {
        "uplift_vs_response": np.empty(n_replicates, dtype=np.float64),
        "uplift_vs_uniform": np.empty(n_replicates, dtype=np.float64),
        "response_vs_uniform": np.empty(n_replicates, dtype=np.float64),
    }

    for b in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        n_b = len(idx)
        k_b = (
            compute_budget_quota(n_b, budget_share=budget_share)
            if quota_override is None
            else int(quota_override)
        )

        z_b = z_full[idx]
        p1_b = p1_arr[idx]
        up_b = up_arr[idx]
        id_b = id_arr[idx]

        pi_u = uniform_policy_probabilities(n_b, k_b)
        pi_r = select_top_k_deterministic(p1_b, id_b, k_b)
        pi_up = select_top_k_deterministic(up_b, id_b, k_b)

        v_u = (1000.0 / n_b) * float(np.sum(pi_u * z_b))
        v_r = (1000.0 / n_b) * float(np.sum(pi_r * z_b))
        v_up = (1000.0 / n_b) * float(np.sum(pi_up * z_b))

        boot_values["uniform"][b] = v_u
        boot_values["response"][b] = v_r
        boot_values["uplift"][b] = v_up

        boot_diffs["uplift_vs_response"][b] = v_up - v_r
        boot_diffs["uplift_vs_uniform"][b] = v_up - v_u
        boot_diffs["response_vs_uniform"][b] = v_r - v_u

    value_intervals = {
        k: (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            float(np.std(arr, ddof=1)) if n_replicates > 1 else 0.0,
        )
        for k, arr in boot_values.items()
    }
    diff_intervals = {
        k: (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            float(np.std(arr, ddof=1)) if n_replicates > 1 else 0.0,
        )
        for k, arr in boot_diffs.items()
    }

    return {
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "boot_values": boot_values,
        "boot_diffs": boot_diffs,
        "value_intervals": value_intervals,
        "diff_intervals": diff_intervals,
    }


def check_score_band_balance(
    treatment: np.ndarray,
    visit: np.ndarray,
    uplift_score: np.ndarray,
    pooled_train_e: float,
    n_bands: int = 5,
) -> pd.DataFrame:
    """Check observed treatment rates and visit lift across uplift score quintiles."""
    df_tmp = pd.DataFrame(
        {
            "treatment": np.asarray(treatment, dtype=np.int8),
            "visit": np.asarray(visit, dtype=np.int8),
            "uplift_score": np.asarray(uplift_score, dtype=np.float64),
        }
    )
    # Deterministic rank-based quintiles so ties never collapse bins
    ranks = df_tmp["uplift_score"].rank(method="first", ascending=True)
    df_tmp["uplift_quintile"] = pd.qcut(ranks, q=n_bands, labels=[f"Q{i}" for i in range(1, n_bands + 1)])

    rows = []
    for q_label, grp in df_tmp.groupby("uplift_quintile", observed=False):
        n_q = len(grp)
        t_rate = float(grp["treatment"].mean())
        se_t = float(np.sqrt(pooled_train_e * (1.0 - pooled_train_e) / max(n_q, 1)))
        z_stat = (t_rate - pooled_train_e) / max(se_t, 1e-12)
        t1 = grp["treatment"] == 1
        t0 = grp["treatment"] == 0
        v_t1 = float(grp.loc[t1, "visit"].mean()) if t1.any() else float("nan")
        v_t0 = float(grp.loc[t0, "visit"].mean()) if t0.any() else float("nan")
        rows.append(
            {
                "uplift_quintile": str(q_label),
                "n_users": int(n_q),
                "min_uplift_score": float(grp["uplift_score"].min()),
                "max_uplift_score": float(grp["uplift_score"].max()),
                "observed_treatment_rate": t_rate,
                "assumed_pooled_e": float(pooled_train_e),
                "treatment_rate_gap": t_rate - float(pooled_train_e),
                "z_score_vs_pooled_e": float(z_stat),
                "visit_rate_treated": v_t1,
                "visit_rate_control": v_t0,
                "observed_visit_lift_pp": 100.0 * (v_t1 - v_t0),
            }
        )
    return pd.DataFrame(rows)


def summarize_propensity_overlap(
    treatment: np.ndarray,
    e_hat: np.ndarray,
    clip_bounds: tuple[float, float] = (0.01, 0.99),
) -> dict:
    """Summarize covariate propensity overlap and inverse-probability weight stability."""
    t = np.asarray(treatment, dtype=np.float64)
    e_raw = np.asarray(e_hat, dtype=np.float64)
    e_clipped = np.clip(e_raw, clip_bounds[0], clip_bounds[1])

    ipw_weights = np.where(t == 1.0, 1.0 / e_clipped, 1.0 / (1.0 - e_clipped))
    # Kish effective sample size: (sum w)^2 / sum(w^2)
    ess_total = float((np.sum(ipw_weights) ** 2) / np.sum(ipw_weights**2))

    return {
        "e_hat_min": float(np.min(e_raw)),
        "e_hat_p01": float(np.percentile(e_raw, 1.0)),
        "e_hat_median": float(np.median(e_raw)),
        "e_hat_p99": float(np.percentile(e_raw, 99.0)),
        "e_hat_max": float(np.max(e_raw)),
        "share_outside_005_095": float(np.mean((e_raw < 0.05) | (e_raw > 0.95))),
        "count_clipped_at_001_099": int(np.sum((e_raw < clip_bounds[0]) | (e_raw > clip_bounds[1]))),
        "max_ipw_weight": float(np.max(ipw_weights)),
        "p99_ipw_weight": float(np.percentile(ipw_weights, 99.0)),
        "kish_effective_sample_size": ess_total,
        "nominal_sample_size": int(len(t)),
        "e_clipped": e_clipped,
    }
