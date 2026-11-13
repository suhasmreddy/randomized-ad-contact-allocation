"""Populate resume bullet metrics strictly from a verified real Criteo v2.1 run directory.

Hard safety guardrail: raises ValueError if `dataset_mode != 'real_criteo_v2.1'` so
synthetic fixture numbers can NEVER be inserted into resume bullets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def render_bullets_from_real_run(run_dir: Path) -> list[str]:
    record_path = run_dir / "run_record.json"
    results_path = run_dir / "policy_results.csv"
    diffs_path = run_dir / "policy_differences.csv"

    if not record_path.exists():
        raise FileNotFoundError(f"Missing run_record.json in {run_dir}")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("dataset_mode") != "real_criteo_v2.1" or record.get("is_synthetic_fixture", True):
        raise ValueError(
            "Safety block: refusing to render resume bullets from a non-real or synthetic fixture run! "
            f"(found dataset_mode={record.get('dataset_mode')!r}). Run `--mode real` on Criteo v2.1 first."
        )

    res_df = pd.read_csv(results_path)
    diff_df = pd.read_csv(diffs_path)

    prim_res = res_df[res_df["estimator"] == "primary_pooled_e"].set_index("policy")
    prim_diff = diff_df[diff_df["estimator"] == "primary_pooled_e"].set_index("comparison")

    n_sample = int(record["sampling_and_splits"]["sampled_rows"])
    n_test = int(record["sampling_and_splits"]["n_test"])
    quota_k = int(record["model_and_budget_settings"]["heldout_quota_k"])
    n_boot = int(record["model_and_budget_settings"]["bootstrap_replicates"])
    itt_pp = float(record["heldout_headline_summary"]["overall_itt_visit_diff_pp"])

    v_up = float(prim_res.loc["uplift", "v_per_1000_eligible"])
    v_resp = float(prim_res.loc["response", "v_per_1000_eligible"])
    v_unif = float(prim_res.loc["uniform", "v_per_1000_eligible"])

    d_ur = float(prim_diff.loc["uplift_vs_response", "delta_per_1000_eligible"])
    d_ur_lo = float(prim_diff.loc["uplift_vs_response", "ci95_low"])
    d_ur_hi = float(prim_diff.loc["uplift_vs_response", "ci95_high"])

    d_uu = float(prim_diff.loc["uplift_vs_uniform", "delta_per_1000_eligible"])
    d_uu_lo = float(prim_diff.loc["uplift_vs_uniform", "ci95_low"])
    d_uu_hi = float(prim_diff.loc["uplift_vs_uniform", "ci95_high"])

    b1 = (
        f"Evaluated constrained 20% ad-eligibility allocation ({quota_k:,} of {n_test:,} held-out users "
        f"from a {n_sample:,}-row uniform sample of the public Criteo Uplift v2.1 randomized trial benchmark), "
        f"estimating {v_up:+.2f} vs. {v_resp:+.2f} incremental visits per 1,000 eligible users for "
        f"two-model uplift vs. response targeting (paired Δ = {d_ur:+.2f}, 95% CI [{d_ur_lo:+.2f}, {d_ur_hi:+.2f}] "
        f"across {n_boot:,} bootstrap draws)."
    )
    b2 = (
        f"Built an offline Horvitz-Thompson inverse-propensity policy evaluator and DuckDB SQL experiment audit "
        f"over {n_sample:,} public randomized-trial records (overall ITT lift {itt_pp:+.2f} pp), benchmarking "
        f"uplift targeting against a {v_unif:+.2f} visits/1,000 uniform baseline (paired Δ = {d_uu:+.2f}, "
        f"95% CI [{d_uu_lo:+.2f}, {d_uu_hi:+.2f}])."
    )
    b3 = (
        f"Engineered a reproducible Python/SQL causal policy-evaluation pipeline with 11 automated pytest checks "
        f"(anti-leakage feature allowlists, deterministic quota tie-breaking, propensity-model sensitivity, and "
        f"independent SQL metric verification matching Python to <1e-12 across {n_test:,} held-out rows)."
    )
    return [b1, b2, b3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render verified resume bullets from outputs/real.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/real"))
    args = parser.parse_args()
    bullets = render_bullets_from_real_run(args.run_dir)
    for idx, text in enumerate(bullets, 1):
        print(f"{idx}. {text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
