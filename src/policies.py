"""Budget quota calculation and deterministic policy selection rules."""

from __future__ import annotations

import math
import numpy as np


def compute_budget_quota(n_users: int, budget_share: float = 0.20) -> int:
    """Compute exact integer ad-eligibility quota K = floor(budget_share * N)."""
    if n_users < 0:
        raise ValueError(f"n_users must be non-negative, got {n_users}.")
    if not (0.0 <= budget_share <= 1.0):
        raise ValueError(f"budget_share must lie in [0, 1], got {budget_share}.")
    return int(math.floor(budget_share * n_users))


def uniform_policy_probabilities(n_users: int, quota_k: int) -> np.ndarray:
    """Return exact expected selection probabilities K / N for every user."""
    if n_users == 0:
        return np.zeros(0, dtype=np.float64)
    if not (0 <= quota_k <= n_users):
        raise ValueError(f"quota_k={quota_k} must satisfy 0 <= quota_k <= n_users={n_users}.")
    prob = float(quota_k) / float(n_users)
    return np.full(n_users, prob, dtype=np.float64)


def select_top_k_deterministic(
    scores: np.ndarray,
    tie_breaker_ids: np.ndarray,
    quota_k: int,
) -> np.ndarray:
    """Select exact K users with highest scores, breaking ties deterministically.

    Sorting priority:
      1. Descending `scores` (-scores ascending)
      2. Ascending `tie_breaker_ids` (e.g., original `source_row_id`)
      3. Ascending within-cohort slot `0..N-1` (guarantees exact quota K even when
         bootstrap resampling with replacement draws identical `(score, source_row_id)` pairs).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    ids_arr = np.asarray(tie_breaker_ids, dtype=np.int64)
    n = len(scores_arr)
    if len(ids_arr) != n:
        raise ValueError("scores and tie_breaker_ids must have identical length.")
    if not (0 <= quota_k <= n):
        raise ValueError(f"quota_k={quota_k} must satisfy 0 <= quota_k <= n={n}.")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("Policy scores contain NaN or infinite values.")

    selected = np.zeros(n, dtype=np.float64)
    if quota_k == 0 or n == 0:
        return selected

    cohort_pos = np.arange(n, dtype=np.int64)
    # np.lexsort sorts by last key primary, second-to-last secondary, first key tertiary
    order = np.lexsort((cohort_pos, ids_arr, -scores_arr))
    selected[order[:quota_k]] = 1.0
    return selected
