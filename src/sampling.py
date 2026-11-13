"""Streaming uniform reservoir sampling, 60/20/20 splitting, and labeled synthetic fixture generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.schema import ALLOWED_FEATURES, BINARY_COLUMNS, EXPECTED_RAW_COLUMNS, validate_raw_schema


def compute_sha256(file_path: Path, block_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def stream_uniform_sample(
    archive_path: Path,
    sample_size: int = 200_000,
    seed: int = 20261112,
    chunksize: int = 250_000,
) -> Tuple[pd.DataFrame, dict]:
    """Draw a reproducible uniform without-replacement sample by streaming chunks.

    Uses priority-key reservoir sampling: each scanned row j across the full archive receives
    an independent U(0, 1) priority drawn sequentially from PCG64(seed), and the `sample_size`
    rows with the smallest keys are retained. Because keys are drawn per row in stream order,
    every row in the 13.98M-row archive has exact inclusion probability `sample_size / N_total`,
    the sample never takes the first `sample_size` rows, and the output is invariant to `chunksize`.
    """
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")

    rng = np.random.default_rng(seed)
    dtype_map = {f: np.float64 for f in ALLOWED_FEATURES}
    for bcol in BINARY_COLUMNS:
        dtype_map[bcol] = np.int8

    reservoir_df: pd.DataFrame | None = None
    reservoir_keys: np.ndarray | None = None
    total_rows_scanned = 0
    chunks_processed = 0

    reader = pd.read_csv(
        archive_path,
        compression="infer",
        chunksize=chunksize,
        dtype=dtype_map,
    )

    for chunk in reader:
        chunks_processed += 1
        if chunks_processed == 1:
            missing = [c for c in EXPECTED_RAW_COLUMNS if c not in chunk.columns]
            if missing:
                raise ValueError(f"Archive missing expected columns: {missing}")
            extra = [c for c in chunk.columns if c not in EXPECTED_RAW_COLUMNS]
            if extra:
                raise ValueError(f"Archive contains unexpected columns: {extra}")

        n_chunk = len(chunk)
        row_ids = np.arange(total_rows_scanned, total_rows_scanned + n_chunk, dtype=np.int64)
        total_rows_scanned += n_chunk

        chunk_keys = rng.random(n_chunk)
        chunk = chunk.loc[:, list(EXPECTED_RAW_COLUMNS)].copy()
        chunk.insert(0, "source_row_id", row_ids)

        if reservoir_df is None:
            if n_chunk <= sample_size:
                reservoir_df = chunk.reset_index(drop=True)
                reservoir_keys = chunk_keys
            else:
                keep_idx = np.argpartition(chunk_keys, sample_size - 1)[:sample_size]
                reservoir_df = chunk.iloc[keep_idx].reset_index(drop=True)
                reservoir_keys = chunk_keys[keep_idx]
        else:
            assert reservoir_keys is not None
            combined_keys = np.concatenate([reservoir_keys, chunk_keys])
            combined_df = pd.concat([reservoir_df, chunk], ignore_index=True)
            if len(combined_df) > sample_size:
                keep_idx = np.argpartition(combined_keys, sample_size - 1)[:sample_size]
                reservoir_df = combined_df.iloc[keep_idx].reset_index(drop=True)
                reservoir_keys = combined_keys[keep_idx]
            else:
                reservoir_df = combined_df
                reservoir_keys = combined_keys

    if reservoir_df is None or total_rows_scanned < sample_size:
        raise ValueError(
            f"Archive had {total_rows_scanned} rows, fewer than requested sample_size={sample_size}."
        )

    # Sort deterministically by original archive row position
    reservoir_df = reservoir_df.sort_values("source_row_id", kind="mergesort").reset_index(drop=True)
    validate_raw_schema(reservoir_df)

    meta = {
        "total_archive_rows_scanned": int(total_rows_scanned),
        "sampled_rows": int(len(reservoir_df)),
        "sampling_fraction": float(len(reservoir_df) / total_rows_scanned),
        "sampling_seed": int(seed),
        "chunksize_used": int(chunksize),
        "chunks_processed": int(chunks_processed),
        "min_source_row_id": int(reservoir_df["source_row_id"].min()),
        "max_source_row_id": int(reservoir_df["source_row_id"].max()),
    }
    return reservoir_df, meta


def assign_splits(
    df: pd.DataFrame,
    split_seed: int = 20261113,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> pd.DataFrame:
    """Assign reproducible 60/20/20 train/val/test splits with disjoint row sets."""
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("Invalid split fractions.")

    n = len(df)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    if n_train <= 0 or n_val <= 0 or (n - n_train - n_val) <= 0:
        raise ValueError(f"Dataset of size {n} too small for 60/20/20 split.")

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)

    split_labels = np.empty(n, dtype=object)
    split_labels[perm[:n_train]] = "train"
    split_labels[perm[n_train : n_train + n_val]] = "val"
    split_labels[perm[n_train + n_val :]] = "test"

    out = df.copy()
    out["split"] = split_labels
    return out


def generate_synthetic_fixture(
    output_path: Path,
    n_rows: int = 20_000,
    seed: int = 20260101,
) -> dict:
    """Create an explicitly labeled synthetic Criteo-schema fixture for offline pipeline verification.

    Includes four realistic behavioral segments under randomized 85/15 assignment:
    - 'Sure Things': high baseline visit probability driven by f0 and f1, near-zero ad lift.
    - 'Persuadables': moderate baseline visit probability, positive incremental lift driven by f2 and f3.
    - 'Lost Causes': very low baseline visit probability and negligible lift.
    - 'Mild Sleeping Dogs': high-baseline segment with slightly negative incremental effect.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_rows, 12))
    # Add realistic skew/shift to several features like projected Criteo features
    X[:, 0] = rng.normal(loc=1.8, scale=1.2, size=n_rows)
    X[:, 1] = np.exp(rng.normal(loc=0.0, scale=0.55, size=n_rows))
    X[:, 5] = rng.laplace(loc=-0.4, scale=0.9, size=n_rows)

    # True randomized eligibility with p(T=1) = 0.85
    treatment = (rng.random(n_rows) < 0.85).astype(np.int8)

    # Post-assignment ad exposure occurs ONLY among treated users, correlated with browsing activity f0/f1
    logit_exposure = -2.2 + 0.65 * X[:, 0] + 0.40 * X[:, 1]
    prob_exposure_given_t1 = 1.0 / (1.0 + np.exp(-logit_exposure))
    exposure = np.where(
        treatment == 1,
        (rng.random(n_rows) < prob_exposure_given_t1).astype(np.int8),
        np.int8(0),
    )

    # Potential visit probability under control: strongly driven by f0 and f1 ("sure things")
    logit_y0 = -3.75 + 0.95 * (X[:, 0] - 1.8) + 0.70 * (X[:, 1] - 1.15) - 0.25 * X[:, 4]
    p_y0 = 1.0 / (1.0 + np.exp(-logit_y0))

    # Heterogeneous incremental effect on log-odds: driven by f2, f3, negatively related to f0 ("sure things" waste)
    tau_logit = 0.32 + 0.75 * X[:, 2] + 0.55 * X[:, 3] - 0.45 * (X[:, 0] - 1.8)
    p_y1 = 1.0 / (1.0 + np.exp(-(logit_y0 + tau_logit)))

    prob_visit = np.where(treatment == 1, p_y1, p_y0)
    visit = (rng.random(n_rows) < prob_visit).astype(np.int8)

    # Secondary label `conversion` (not used in primary visit evaluation, present in Criteo schema)
    prob_conv = np.where(visit == 1, np.clip(0.06 + 0.02 * treatment, 0.01, 0.25), 0.0)
    conversion = (rng.random(n_rows) < prob_conv).astype(np.int8)

    data = {f"f{i}": np.round(X[:, i], 6) for i in range(12)}
    data["treatment"] = treatment
    data["conversion"] = conversion
    data["visit"] = visit
    data["exposure"] = exposure

    df = pd.DataFrame(data, columns=list(EXPECTED_RAW_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, compression="gzip")

    return {
        "fixture_path": str(output_path),
        "n_rows": int(n_rows),
        "seed": int(seed),
        "sha256": compute_sha256(output_path),
        "treatment_rate": float(df["treatment"].mean()),
        "visit_rate": float(df["visit"].mean()),
    }
