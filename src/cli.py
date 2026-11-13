"""End-to-end CLI entrypoint for Randomized Ad-Contact Allocation.

Single-command fixture run:
    python3 -m src.cli --mode fixture

Single-command full 200,000-row real Criteo v2.1 run:
    python3 -m src.cli --mode real --data-path data/raw/criteo-research-uplift-v2.1.csv.gz
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

from src.audit import get_sql_engine_info, recompute_headline_metrics_sql, run_sql_audit
from src.evaluation import (
    check_score_band_balance,
    compute_overall_itt_visit_diff_pp,
    evaluate_three_policies,
    paired_bootstrap_evaluation,
    summarize_propensity_overlap,
)
from src.modeling import (
    compute_validation_diagnostics,
    fit_two_model_uplift_and_propensity,
    get_ml_backend_info,
)
from src.plotting import plot_policy_comparison
from src.sampling import (
    assign_splits,
    compute_sha256,
    generate_synthetic_fixture,
    stream_uniform_sample,
)
from src.schema import (
    ALLOWED_FEATURES,
    CRITEO_V21_BYTES,
    CRITEO_V21_FILENAME,
    CRITEO_V21_SHA256,
    CRITEO_V21_TOTAL_ROWS,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _classify_winner(ci_low: float, ci_high: float) -> str:
    if ci_low > 0.0:
        return "positive_statistically_distinguishable"
    if ci_high < 0.0:
        return "negative_statistically_distinguishable"
    return "inconclusive_spans_zero"


def run_pipeline(
    mode: str,
    data_path: Path,
    output_dir: Path,
    sample_size: int,
    budget_share: float,
    regularization_c: float,
    bootstrap_replicates: int,
    sample_seed: int,
    split_seed: int,
    model_seed: int,
    bootstrap_seed: int,
    strict_hash_check: bool = True,
) -> dict:
    """Execute full experiment audit, T-learner fit, IPW evaluation, sensitivity, and SQL cross-check."""
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "fixture":
        dataset_mode = "synthetic_fixture"
        if not data_path.exists():
            generate_synthetic_fixture(data_path, n_rows=sample_size, seed=20260101)
        file_sha256 = compute_sha256(data_path)
        file_bytes = data_path.stat().st_size
        sampled_df, sample_meta = stream_uniform_sample(
            data_path,
            sample_size=sample_size,
            seed=sample_seed,
            chunksize=5_000,
        )
        hash_verified = True
    elif mode == "real":
        dataset_mode = "real_criteo_v2.1"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Real Criteo archive not found at '{data_path}'.\n"
                f"Place '{CRITEO_V21_FILENAME}' at '{data_path}' (or run `python3 scripts/fetch_criteo_v21.py`),\n"
                f"then re-run:\n"
                f"  python3 -m src.cli --mode real --data-path {data_path}"
            )
        file_bytes = data_path.stat().st_size
        file_sha256 = compute_sha256(data_path)
        hash_verified = file_sha256 == CRITEO_V21_SHA256
        if strict_hash_check and not hash_verified:
            raise ValueError(
                f"SHA-256 mismatch for {data_path}!\n"
                f"  Expected (Criteo v2.1): {CRITEO_V21_SHA256}\n"
                f"  Observed:               {file_sha256}\n"
                f"Verify you downloaded the corrected v2.1 archive ({CRITEO_V21_BYTES:,} bytes), not v1."
            )
        sampled_df, sample_meta = stream_uniform_sample(
            data_path,
            sample_size=sample_size,
            seed=sample_seed,
            chunksize=250_000,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 1. Assign 60/20/20 splits and run SQL Experiment Audit
    df_splits = assign_splits(sampled_df, split_seed=split_seed, train_frac=0.60, val_frac=0.20)
    audit_sql_path = REPO_ROOT / "sql" / "experiment_audit.sql"
    split_audit_df, balance_audit_df = run_sql_audit(df_splits, audit_sql_path)
    split_audit_df.insert(0, "dataset_mode", dataset_mode)
    balance_audit_df.insert(0, "dataset_mode", dataset_mode)
    split_audit_df.to_csv(output_dir / "audit_splits.csv", index=False)
    balance_audit_df.to_csv(output_dir / "audit_feature_balance.csv", index=False)

    # 2. Fit two L2-regularized logistic models + propensity model on TRAIN only
    train_df = df_splits[df_splits["split"] == "train"].reset_index(drop=True)
    val_df = df_splits[df_splits["split"] == "val"].reset_index(drop=True)
    test_df = df_splits[df_splits["split"] == "test"].reset_index(drop=True)

    models = fit_two_model_uplift_and_propensity(
        train_df,
        C=regularization_c,
        seed=model_seed,
        feature_cols=ALLOWED_FEATURES,
    )

    val_diag_df = compute_validation_diagnostics(models, val_df)
    val_diag_df.insert(0, "dataset_mode", dataset_mode)
    val_diag_df.to_csv(output_dir / "validation_diagnostics.csv", index=False)

    # 3. Held-out test evaluation (frozen model settings & frozen 20% budget)
    test_preds = models.predict_outcomes_and_uplift(test_df)
    t_test = test_df["treatment"].to_numpy(dtype=np.int8)
    y_test = test_df["visit"].to_numpy(dtype=np.int8)
    ids_test = test_df["source_row_id"].to_numpy(dtype=np.int64)

    itt_summary = compute_overall_itt_visit_diff_pp(t_test, y_test)

    # Primary estimator: pooled training treatment fraction e
    primary_eval = evaluate_three_policies(
        treatment=t_test,
        visit=y_test,
        p_treat=test_preds["p_treat"],
        uplift_score=test_preds["uplift_score"],
        tie_breaker_ids=ids_test,
        propensity=models.pooled_train_e,
        budget_share=budget_share,
    )
    primary_boot = paired_bootstrap_evaluation(
        treatment=t_test,
        visit=y_test,
        p_treat=test_preds["p_treat"],
        uplift_score=test_preds["uplift_score"],
        tie_breaker_ids=ids_test,
        propensity=models.pooled_train_e,
        budget_share=budget_share,
        n_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    # Score-band randomization balance check
    band_df = check_score_band_balance(
        treatment=t_test,
        visit=y_test,
        uplift_score=test_preds["uplift_score"],
        pooled_train_e=models.pooled_train_e,
        n_bands=5,
    )
    band_df.insert(0, "dataset_mode", dataset_mode)
    band_df.to_csv(output_dir / "score_band_balance.csv", index=False)

    # Sensitivity check: covariate-adjusted training propensity e(X)
    overlap_summary = summarize_propensity_overlap(t_test, test_preds["e_hat"])
    e_x_clipped = overlap_summary.pop("e_clipped")

    sens_eval = evaluate_three_policies(
        treatment=t_test,
        visit=y_test,
        p_treat=test_preds["p_treat"],
        uplift_score=test_preds["uplift_score"],
        tie_breaker_ids=ids_test,
        propensity=e_x_clipped,
        budget_share=budget_share,
    )
    sens_boot = paired_bootstrap_evaluation(
        treatment=t_test,
        visit=y_test,
        p_treat=test_preds["p_treat"],
        uplift_score=test_preds["uplift_score"],
        tie_breaker_ids=ids_test,
        propensity=e_x_clipped,
        budget_share=budget_share,
        n_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    # Assemble policy_results.csv and policy_differences.csv
    res_rows = []
    for est_name, ev, bt in [
        ("primary_pooled_e", primary_eval, primary_boot),
        ("sensitivity_propensity_e_x", sens_eval, sens_boot),
    ]:
        for pol in ["uniform", "response", "uplift"]:
            ci_lo, ci_hi, se = bt["value_intervals"][pol]
            res_rows.append(
                {
                    "dataset_mode": dataset_mode,
                    "estimator": est_name,
                    "policy": pol,
                    "n_heldout_eligible": ev["n_users"],
                    "quota_k": ev["quota_k"],
                    "budget_share": ev["budget_share_realized"],
                    "v_per_1000_eligible": ev["values_per_1k"][pol],
                    "bootstrap_se": se,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "bootstrap_replicates": bt["n_replicates"],
                }
            )
    results_df = pd.DataFrame(res_rows)
    results_df.to_csv(output_dir / "policy_results.csv", index=False)

    diff_rows = []
    for est_name, ev, bt in [
        ("primary_pooled_e", primary_eval, primary_boot),
        ("sensitivity_propensity_e_x", sens_eval, sens_boot),
    ]:
        for comp, policy_a, policy_b in [
            ("uplift_vs_response", "uplift", "response"),
            ("uplift_vs_uniform", "uplift", "uniform"),
            ("response_vs_uniform", "response", "uniform"),
        ]:
            ci_lo, ci_hi, se = bt["diff_intervals"][comp]
            delta_val = ev["differences_per_1k"][comp]
            diff_rows.append(
                {
                    "dataset_mode": dataset_mode,
                    "estimator": est_name,
                    "comparison": comp,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "delta_per_1000_eligible": delta_val,
                    "bootstrap_se": se,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "ci95_excludes_zero": bool(ci_lo > 0.0 or ci_hi < 0.0),
                    "conclusion": _classify_winner(ci_lo, ci_hi),
                    "bootstrap_replicates": bt["n_replicates"],
                }
            )
    diffs_df = pd.DataFrame(diff_rows)
    diffs_df.to_csv(output_dir / "policy_differences.csv", index=False)

    # Export held-out evaluation rows and verify headline numbers independently via SQL
    eval_rows_df = pd.DataFrame(
        {
            "source_row_id": ids_test,
            "f0": test_df["f0"].to_numpy(),
            "f1": test_df["f1"].to_numpy(),
            "f2": test_df["f2"].to_numpy(),
            "f3": test_df["f3"].to_numpy(),
            "treatment": t_test,
            "visit": y_test,
            "e_pooled": np.full(len(t_test), models.pooled_train_e, dtype=np.float64),
            "e_hat_x": e_x_clipped,
            "p_treat": test_preds["p_treat"],
            "p_ctrl": test_preds["p_ctrl"],
            "uplift_score": test_preds["uplift_score"],
            "pi_uniform": primary_eval["pi_uniform"],
            "pi_response": primary_eval["pi_response"],
            "pi_uplift": primary_eval["pi_uplift"],
            "z_pooled": primary_eval["z"],
            "contrib_uniform_per_1k": (1000.0 / len(t_test)) * primary_eval["pi_uniform"] * primary_eval["z"],
            "contrib_response_per_1k": (1000.0 / len(t_test)) * primary_eval["pi_response"] * primary_eval["z"],
            "contrib_uplift_per_1k": (1000.0 / len(t_test)) * primary_eval["pi_uplift"] * primary_eval["z"],
        }
    )
    eval_csv_path = output_dir / "heldout_eval_rows.csv"
    eval_rows_df.to_csv(eval_csv_path, index=False)

    verify_sql_path = REPO_ROOT / "sql" / "verify_headline_metrics.sql"
    sql_check = recompute_headline_metrics_sql(
        eval_csv_path=eval_csv_path,
        sql_path=verify_sql_path,
        python_metrics={
            "v_uniform_per_1k": primary_eval["values_per_1k"]["uniform"],
            "v_response_per_1k": primary_eval["values_per_1k"]["response"],
            "v_uplift_per_1k": primary_eval["values_per_1k"]["uplift"],
            "delta_uplift_vs_response_per_1k": primary_eval["differences_per_1k"]["uplift_vs_response"],
            "delta_uplift_vs_uniform_per_1k": primary_eval["differences_per_1k"]["uplift_vs_uniform"],
            "delta_response_vs_uniform_per_1k": primary_eval["differences_per_1k"]["response_vs_uniform"],
            "overall_itt_visit_diff_pp": itt_summary["itt_visit_diff_pp"],
        },
    )
    if not sql_check["all_metrics_agree"]:
        raise RuntimeError(f"Independent SQL recomputation disagreed with Python evaluator: {sql_check}")

    (output_dir / "sql_recomputation_check.json").write_text(
        json.dumps(sql_check, indent=2),
        encoding="utf-8",
    )

    # Check whether ranking changed under propensity sensitivity e(X)
    primary_ranking = sorted(
        ["uniform", "response", "uplift"],
        key=lambda p: primary_eval["values_per_1k"][p],
        reverse=True,
    )
    sensitivity_ranking = sorted(
        ["uniform", "response", "uplift"],
        key=lambda p: sens_eval["values_per_1k"][p],
        reverse=True,
    )
    ranking_changed = primary_ranking != sensitivity_ranking

    # Plot comparison chart
    chart_path = output_dir / "policy_comparison.png"
    plot_policy_comparison(
        results_df=results_df,
        diffs_df=diffs_df,
        output_path=chart_path,
        dataset_mode=dataset_mode,
    )

    elapsed_sec = time.perf_counter() - t_start
    repro_cmd = (
        "python3 -m src.cli --mode fixture"
        if mode == "fixture"
        else f"python3 -m src.cli --mode real --data-path {data_path}"
    )

    run_record = {
        "project": "Randomized Ad-Contact Allocation",
        "dataset_mode": dataset_mode,
        "is_synthetic_fixture": mode == "fixture",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction_command": repro_cmd,
        "runtime_seconds": round(elapsed_sec, 3),
        "source_metadata": {
            "publisher": "Criteo AI Lab (Diemert, Betlei, Renaudin, Amini, AdKDD 2018)",
            "publisher_url": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
            "mirror_url": "https://huggingface.co/datasets/criteo/criteo-uplift/tree/main",
            "license": "CC BY-NC-SA 4.0",
            "input_file_path": str(data_path),
            "input_file_bytes": int(file_bytes),
            "input_file_sha256": file_sha256,
            "expected_criteo_v21_sha256": CRITEO_V21_SHA256,
            "expected_criteo_v21_rows": CRITEO_V21_TOTAL_ROWS,
            "sha256_matches_official_v21": bool(hash_verified) if mode == "real" else False,
        },
        "sampling_and_splits": {
            **sample_meta,
            "split_seed": int(split_seed),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
        "model_and_budget_settings": {
            "allowed_features": list(ALLOWED_FEATURES),
            "excluded_columns": ["exposure", "visit", "conversion", "treatment", "source_row_id"],
            "model_family": "Two-model L2-regularized Logistic Regression (T-learner)",
            "regularization_C": float(regularization_c),
            "model_seed": int(model_seed),
            "budget_share": float(budget_share),
            "heldout_quota_k": int(primary_eval["quota_k"]),
            "pooled_train_treatment_rate_e": float(models.pooled_train_e),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "sql_engine": get_sql_engine_info(),
            "ml_backend": get_ml_backend_info(),
        },
        "heldout_headline_summary": {
            "overall_itt_visit_rate_treated": itt_summary["visit_rate_treated"],
            "overall_itt_visit_rate_control": itt_summary["visit_rate_control"],
            "overall_itt_visit_diff_pp": itt_summary["itt_visit_diff_pp"],
            "primary_values_per_1k": primary_eval["values_per_1k"],
            "primary_differences_per_1k": primary_eval["differences_per_1k"],
            "primary_ranking": primary_ranking,
            "sensitivity_values_per_1k": sens_eval["values_per_1k"],
            "sensitivity_differences_per_1k": sens_eval["differences_per_1k"],
            "sensitivity_ranking": sensitivity_ranking,
            "ranking_changed_under_e_x_sensitivity": bool(ranking_changed),
            "propensity_overlap": overlap_summary,
            "sql_recomputation_all_agree": bool(sql_check["all_metrics_agree"]),
        },
    }

    (output_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2),
        encoding="utf-8",
    )
    return run_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomized Ad-Contact Allocation: Audit, T-Learner, and IPW Policy Evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["fixture", "real"],
        required=True,
        help="'fixture' runs the labeled synthetic fixture; 'real' runs the 200,000-row Criteo v2.1 sample.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Input .csv.gz path (defaults based on --mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to outputs/fixture or outputs/real).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Uniform sample size (defaults to 20,000 in fixture mode; fixed at 200,000 in real mode).",
    )
    parser.add_argument("--budget-share", type=float, default=0.20)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20261112)
    parser.add_argument("--split-seed", type=int, default=20261113)
    parser.add_argument("--model-seed", type=int, default=20261114)
    parser.add_argument("--bootstrap-seed", type=int, default=20261115)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Allow non-matching SHA-256 in real mode (not recommended).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "fixture":
        data_path = args.data_path or (REPO_ROOT / "data" / "fixtures" / "synthetic_criteo_fixture.csv.gz")
        output_dir = args.output_dir or (REPO_ROOT / "outputs" / "fixture")
        sample_size = args.sample_size or 20_000
    else:
        data_path = args.data_path or (REPO_ROOT / "data" / "raw" / CRITEO_V21_FILENAME)
        output_dir = args.output_dir or (REPO_ROOT / "outputs" / "real")
        sample_size = args.sample_size or 200_000

    record = run_pipeline(
        mode=args.mode,
        data_path=data_path,
        output_dir=output_dir,
        sample_size=sample_size,
        budget_share=args.budget_share,
        regularization_c=args.regularization_c,
        bootstrap_replicates=args.bootstrap_replicates,
        sample_seed=args.sample_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        bootstrap_seed=args.bootstrap_seed,
        strict_hash_check=not args.skip_hash_check,
    )

    hl = record["heldout_headline_summary"]
    print(f"[{record['dataset_mode']}] Completed in {record['runtime_seconds']:.2f}s -> {output_dir}")
    print(
        f"  Held-out N={record['sampling_and_splits']['n_test']:,}, "
        f"Quota K={record['model_and_budget_settings']['heldout_quota_k']:,}, "
        f"Overall ITT visit diff={hl['overall_itt_visit_diff_pp']:+.4f} pp"
    )
    for pol, val in hl["primary_values_per_1k"].items():
        print(f"  V({pol:<8s}) = {val:+.4f} incremental visits / 1,000 eligible")
    for comp, val in hl["primary_differences_per_1k"].items():
        print(f"  Delta({comp:<20s}) = {val:+.4f} / 1,000 eligible")
    print(f"  SQL recomputation agreement: {hl['sql_recomputation_all_agree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
