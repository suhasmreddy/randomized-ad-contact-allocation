"""SQL experiment audit and independent headline metric recomputation.

Uses DuckDB when installed (`pip install -r requirements.txt`), with an automatic
stdlib `sqlite3` fallback for offline sandbox environments lacking `duckdb`.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd

from src.schema import ALLOWED_FEATURES

try:
    import duckdb  # type: ignore

    HAS_DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore
    HAS_DUCKDB = False


def get_sql_engine_info() -> str:
    """Return active SQL engine name and version for the run record."""
    if HAS_DUCKDB:
        return f"duckdb {duckdb.__version__}"
    return f"sqlite3 {sqlite3.sqlite_version} (stdlib fallback; install duckdb for native DuckDB execution)"


def _execute_sql_on_df(sql_text: str, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Execute an ANSI SQL query against a single registered DataFrame."""
    if HAS_DUCKDB:
        con = duckdb.connect(database=":memory:")
        try:
            con.register(table_name, df)
            return con.execute(sql_text).df()
        finally:
            con.close()

    con = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, con, index=False, if_exists="replace")
        return pd.read_sql_query(sql_text, con)
    finally:
        con.close()


def run_sql_audit(
    df_with_splits: pd.DataFrame,
    sql_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run SQL split/arm/outcome/missingness audit and compute f0..f11 balance via SQL."""
    sql_text = sql_path.read_text(encoding="utf-8")
    split_summary = _execute_sql_on_df(sql_text, "sample_data", df_with_splits)

    # Order splits cleanly: train, val, test, ALL
    order_map = {"train": 0, "val": 1, "test": 2, "ALL": 3}
    split_summary["_ord"] = split_summary["split"].map(lambda s: order_map.get(s, 99))
    split_summary = split_summary.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    # Feature balance query across f0..f11 (overall sample and train/test check)
    balance_queries = []
    for feat in ALLOWED_FEATURES:
        balance_queries.append(
            f"""
            SELECT
                '{feat}' AS feature,
                COUNT(*) AS n_rows,
                SUM(CASE WHEN {feat} IS NULL THEN 1 ELSE 0 END) AS null_count,
                AVG(CASE WHEN treatment = 1 THEN CAST({feat} AS DOUBLE) END) AS mean_treated,
                AVG(CASE WHEN treatment = 0 THEN CAST({feat} AS DOUBLE) END) AS mean_control,
                AVG(CASE WHEN treatment = 1 THEN CAST({feat} AS DOUBLE) * CAST({feat} AS DOUBLE) END)
                    - AVG(CASE WHEN treatment = 1 THEN CAST({feat} AS DOUBLE) END)
                    * AVG(CASE WHEN treatment = 1 THEN CAST({feat} AS DOUBLE) END) AS var_treated_pop,
                AVG(CASE WHEN treatment = 0 THEN CAST({feat} AS DOUBLE) * CAST({feat} AS DOUBLE) END)
                    - AVG(CASE WHEN treatment = 0 THEN CAST({feat} AS DOUBLE) END)
                    * AVG(CASE WHEN treatment = 0 THEN CAST({feat} AS DOUBLE) END) AS var_control_pop
            FROM sample_data
            """
        )
    balance_sql = "\nUNION ALL\n".join(balance_queries)
    balance_df = _execute_sql_on_df(balance_sql, "sample_data", df_with_splits)

    mean_diff = balance_df["mean_treated"] - balance_df["mean_control"]
    pooled_sd = np.sqrt(
        np.maximum(
            0.5 * (balance_df["var_treated_pop"].clip(lower=0.0) + balance_df["var_control_pop"].clip(lower=0.0)),
            1e-12,
        )
    )
    balance_df["mean_diff"] = mean_diff
    balance_df["pooled_sd"] = pooled_sd
    balance_df["smd"] = mean_diff / pooled_sd
    balance_df["abs_smd"] = balance_df["smd"].abs()
    balance_df = balance_df[
        [
            "feature",
            "n_rows",
            "null_count",
            "mean_treated",
            "mean_control",
            "mean_diff",
            "pooled_sd",
            "smd",
            "abs_smd",
        ]
    ]
    return split_summary, balance_df


def recompute_headline_metrics_sql(
    eval_csv_path: Path,
    sql_path: Path,
    python_metrics: dict[str, float],
    atol: float = 1e-9,
) -> dict:
    """Independently recompute headline policy values via SQL against exported CSV rows."""
    exported_df = pd.read_csv(eval_csv_path)
    sql_text = sql_path.read_text(encoding="utf-8")
    sql_df = _execute_sql_on_df(sql_text, "heldout_eval_rows", exported_df)
    row = sql_df.iloc[0].to_dict()

    comparisons = {}
    all_agree = True
    for key, py_val in python_metrics.items():
        sql_val = float(row[key])
        abs_diff = abs(float(py_val) - sql_val)
        agrees = bool(abs_diff <= atol)
        if not agrees:
            all_agree = False
        comparisons[key] = {
            "python_value": float(py_val),
            "sql_value": sql_val,
            "abs_diff": abs_diff,
            "agrees_within_tol": agrees,
        }

    return {
        "sql_engine": get_sql_engine_info(),
        "tolerance": atol,
        "all_metrics_agree": all_agree,
        "comparisons": comparisons,
    }
