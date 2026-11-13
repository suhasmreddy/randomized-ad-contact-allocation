"""Schema constants and anti-leakage feature allowlist enforcement."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

EXPECTED_RAW_COLUMNS: tuple[str, ...] = (
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "treatment",
    "conversion",
    "visit",
    "exposure",
)

ALLOWED_FEATURES: tuple[str, ...] = (
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
)

FORBIDDEN_MODEL_INPUTS: frozenset[str] = frozenset(
    {
        "treatment",
        "visit",
        "conversion",
        "exposure",
        "source_row_id",
        "row_id",
        "split",
    }
)

BINARY_COLUMNS: tuple[str, ...] = ("treatment", "conversion", "visit", "exposure")

# Official Criteo Uplift v2.1 metadata (Hugging Face criteo/criteo-uplift main)
CRITEO_V21_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
CRITEO_V21_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
CRITEO_V21_BYTES = 311_422_618
CRITEO_V21_TOTAL_ROWS = 13_979_592


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Verify exact Criteo Uplift v2.1 column names and binary outcome/assignment values."""
    missing = [col for col in EXPECTED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required Criteo v2.1 columns: {missing}")

    for col in ALLOWED_FEATURES:
        if df[col].isna().any():
            raise ValueError(f"Feature column '{col}' contains null values.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Feature column '{col}' must be numeric, got {df[col].dtype}.")

    for col in BINARY_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Binary column '{col}' contains null values.")
        unique_vals = set(np.unique(df[col].to_numpy()))
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Column '{col}' must contain only binary {{0, 1}} values; got {sorted(unique_vals)}."
            )


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = ALLOWED_FEATURES,
) -> np.ndarray:
    """Extract strictly verified pre-assignment features f0..f11 as float64 array.

    Raises ValueError if any forbidden post-assignment, outcome, treatment, or ID column
    is requested or if any column falls outside ALLOWED_FEATURES.
    """
    requested = tuple(feature_cols)
    if not requested:
        raise ValueError("feature_cols cannot be empty.")

    forbidden_hit = [c for c in requested if c in FORBIDDEN_MODEL_INPUTS]
    if forbidden_hit:
        raise ValueError(
            f"Leakage guard triggered: forbidden columns requested as model inputs: {forbidden_hit}"
        )

    unapproved = [c for c in requested if c not in ALLOWED_FEATURES]
    if unapproved:
        raise ValueError(
            f"Unapproved feature columns requested (only {ALLOWED_FEATURES} allowed): {unapproved}"
        )

    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise ValueError(f"Requested features missing from DataFrame: {missing}")

    matrix = df.loc[:, list(requested)].to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return matrix
