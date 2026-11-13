"""Comprehensive verification suite for Randomized Ad-Contact Allocation.

Compatible with BOTH `pytest` (`pytest -q`) and Python's standard-library
`unittest` (`python3 -m unittest discover -s tests -p "test_*.py"`).
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_resume_bullets import render_bullets_from_real_run
from src.audit import recompute_headline_metrics_sql
from src.cli import run_pipeline
from src.evaluation import (
    compute_ipw_pseudo_outcome,
    compute_policy_value_per_1k,
    evaluate_three_policies,
    paired_bootstrap_evaluation,
)
from src.modeling import fit_two_model_uplift_and_propensity
from src.policies import (
    compute_budget_quota,
    select_top_k_deterministic,
    uniform_policy_probabilities,
)
from src.sampling import assign_splits, generate_synthetic_fixture, stream_uniform_sample
from src.schema import ALLOWED_FEATURES, extract_feature_matrix


class TestAllocationSuite(unittest.TestCase):
    """11 unit and integration checks covering math, boundaries, leakage, bootstrap, and SQL."""

    def test_01_four_user_toy_case_hand_calculated(self) -> None:
        """Hand-worked 4-user toy case with K = 1, a control visitor, and a treated visitor.

        Cohort (N = 4, K = 1):
          - User 0 ('Sure Thing', Control Visitor):   T=0, Y=1, p_treat=0.90, p_ctrl=0.88 => uplift=+0.02
          - User 1 ('Persuadable', Treated Visitor):  T=1, Y=1, p_treat=0.55, p_ctrl=0.15 => uplift=+0.40
          - User 2 ('Lost Cause', Treated Non-Visit): T=1, Y=0, p_treat=0.10, p_ctrl=0.08 => uplift=+0.02
          - User 3 ('Sleeping Dog', Ctrl Non-Visit):  T=0, Y=0, p_treat=0.20, p_ctrl=0.25 => uplift=-0.05

        Case A (e = 0.50):
          z = [-2.0, +2.0, 0.0, 0.0]
          V(response) = (1000/4) * (-2.0) = -500.0
          V(uplift)   = (1000/4) * (+2.0) = +500.0
          V(uniform)  = (1000/4) * 0.25 * (-2 + 2) = 0.0
          Delta(uplift, response) = +1000.0; Delta(uplift, uniform) = +500.0

        Case B (e = 0.80, mirroring asymmetric ~85/15 trial assignment):
          z = [-5.0, +1.25, 0.0, 0.0]
          V(response) = -1250.0; V(uplift) = +312.5; V(uniform) = -234.375
          Delta(uplift, response) = +1562.5; Delta(uplift, uniform) = +546.875
        """
        treatment = np.array([0, 1, 1, 0], dtype=np.int8)
        visit = np.array([1, 1, 0, 0], dtype=np.int8)
        p_treat = np.array([0.90, 0.55, 0.10, 0.20], dtype=np.float64)
        p_ctrl = np.array([0.88, 0.15, 0.08, 0.25], dtype=np.float64)
        uplift = p_treat - p_ctrl
        ids = np.array([10, 11, 12, 13], dtype=np.int64)

        res_50 = evaluate_three_policies(
            treatment=treatment,
            visit=visit,
            p_treat=p_treat,
            uplift_score=uplift,
            tie_breaker_ids=ids,
            propensity=0.50,
            quota_override=1,
        )
        assert np.allclose(res_50["z"], [-2.0, 2.0, 0.0, 0.0])
        assert np.array_equal(res_50["pi_response"], [1.0, 0.0, 0.0, 0.0])
        assert np.array_equal(res_50["pi_uplift"], [0.0, 1.0, 0.0, 0.0])
        assert np.isclose(res_50["values_per_1k"]["response"], -500.0)
        assert np.isclose(res_50["values_per_1k"]["uplift"], 500.0)
        assert np.isclose(res_50["values_per_1k"]["uniform"], 0.0)
        assert np.isclose(res_50["differences_per_1k"]["uplift_vs_response"], 1000.0)
        assert np.isclose(res_50["differences_per_1k"]["uplift_vs_uniform"], 500.0)

        res_80 = evaluate_three_policies(
            treatment=treatment,
            visit=visit,
            p_treat=p_treat,
            uplift_score=uplift,
            tie_breaker_ids=ids,
            propensity=0.80,
            quota_override=1,
        )
        assert np.allclose(res_80["z"], [-5.0, 1.25, 0.0, 0.0])
        assert np.isclose(res_80["values_per_1k"]["response"], -1250.0)
        assert np.isclose(res_80["values_per_1k"]["uplift"], 312.5)
        assert np.isclose(res_80["values_per_1k"]["uniform"], -234.375)
        assert np.isclose(res_80["differences_per_1k"]["uplift_vs_response"], 1562.5)
        assert np.isclose(res_80["differences_per_1k"]["uplift_vs_uniform"], 546.875)

    def test_02_control_term_sign_is_strictly_negative(self) -> None:
        z = compute_ipw_pseudo_outcome(
            treatment=np.array([0, 1, 0, 1]),
            visit=np.array([1, 1, 0, 0]),
            propensity=0.85,
        )
        assert z[0] < 0.0 and np.isclose(z[0], -1.0 / 0.15)
        assert z[1] > 0.0 and np.isclose(z[1], 1.0 / 0.85)
        assert z[2] == 0.0 and z[3] == 0.0

    def test_03_full_cohort_denominator_not_selected_quota_k(self) -> None:
        pi = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        z = np.ones(10, dtype=np.float64)
        val = compute_policy_value_per_1k(pi, z)
        assert np.isclose(val, 200.0)

    def test_04_uniform_policy_expectation_identity(self) -> None:
        rng = np.random.default_rng(42)
        n = 53
        k = compute_budget_quota(n, budget_share=0.20)
        assert k == 10
        t = (rng.random(n) < 0.85).astype(np.int8)
        y = (rng.random(n) < 0.12).astype(np.int8)
        z = compute_ipw_pseudo_outcome(t, y, propensity=0.85)

        pi_unif = uniform_policy_probabilities(n, k)
        pi_all = np.ones(n, dtype=np.float64)
        v_unif = compute_policy_value_per_1k(pi_unif, z)
        v_all = compute_policy_value_per_1k(pi_all, z)
        assert np.isclose(v_unif, (k / n) * v_all, atol=1e-12)

    def test_05_exact_quota_and_deterministic_tie_breaking(self) -> None:
        scores = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50])
        ids = np.array([105, 100, 103, 100, 101, 102])
        sel_k3 = select_top_k_deterministic(scores, ids, quota_k=3)
        assert int(sel_k3.sum()) == 3
        assert np.array_equal(sel_k3, [0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        assert np.array_equal(sel_k3, select_top_k_deterministic(scores, ids, quota_k=3))

    def test_06_k_equals_zero_boundary(self) -> None:
        k = compute_budget_quota(4, budget_share=0.20)
        assert k == 0
        res = evaluate_three_policies(
            treatment=np.array([1, 0, 1, 0]),
            visit=np.array([1, 1, 0, 1]),
            p_treat=np.array([0.8, 0.7, 0.2, 0.1]),
            uplift_score=np.array([0.3, -0.1, 0.2, 0.0]),
            tie_breaker_ids=np.array([1, 2, 3, 4]),
            propensity=0.5,
            budget_share=0.20,
        )
        assert res["quota_k"] == 0
        for pol in ["uniform", "response", "uplift"]:
            assert res["values_per_1k"][pol] == 0.0
        for comp in ["uplift_vs_response", "uplift_vs_uniform", "response_vs_uniform"]:
            assert res["differences_per_1k"][comp] == 0.0

    def test_07_feature_allowlist_blocks_leakage(self) -> None:
        df = pd.DataFrame({col: [0.1, 0.2] for col in ALLOWED_FEATURES})
        for col in ["exposure", "visit", "conversion", "treatment", "source_row_id"]:
            df[col] = [1, 0]

        X = extract_feature_matrix(df)
        assert X.shape == (2, 12)

        for forbidden in ["exposure", "visit", "conversion", "treatment", "source_row_id"]:
            with self.assertRaises(ValueError):
                extract_feature_matrix(df, list(ALLOWED_FEATURES) + [forbidden])

    def test_08_split_separation_and_streaming_sampler_invariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "toy_fixture.csv.gz"
            generate_synthetic_fixture(tmp_path, n_rows=1000, seed=99)

            sample_a, meta_a = stream_uniform_sample(tmp_path, sample_size=200, seed=12345, chunksize=150)
            sample_b, _ = stream_uniform_sample(tmp_path, sample_size=200, seed=12345, chunksize=400)

            assert np.array_equal(sample_a["source_row_id"].to_numpy(), sample_b["source_row_id"].to_numpy())
            assert meta_a["total_archive_rows_scanned"] == 1000
            assert meta_a["max_source_row_id"] > 900

            splits = assign_splits(sample_a, split_seed=777)
            train_ids = set(splits.loc[splits["split"] == "train", "source_row_id"])
            val_ids = set(splits.loc[splits["split"] == "val", "source_row_id"])
            test_ids = set(splits.loc[splits["split"] == "test", "source_row_id"])

            assert len(train_ids) == 120 and len(val_ids) == 40 and len(test_ids) == 40
            assert train_ids.isdisjoint(val_ids)
            assert train_ids.isdisjoint(test_ids)
            assert val_ids.isdisjoint(test_ids)

            with self.assertRaises(ValueError):
                fit_two_model_uplift_and_propensity(splits)

    def test_09_paired_bootstrap_shared_resamples(self) -> None:
        rng = np.random.default_rng(7)
        n = 120
        t = (rng.random(n) < 0.85).astype(np.int8)
        y = (rng.random(n) < 0.15).astype(np.int8)
        shared_score = rng.normal(size=n)
        ids = np.arange(n, dtype=np.int64)

        boot = paired_bootstrap_evaluation(
            treatment=t,
            visit=y,
            p_treat=shared_score,
            uplift_score=shared_score,
            tie_breaker_ids=ids,
            propensity=0.85,
            budget_share=0.20,
            n_replicates=50,
            seed=999,
        )
        assert np.all(boot["boot_diffs"]["uplift_vs_response"] == 0.0)
        lhs = boot["boot_diffs"]["uplift_vs_uniform"] - boot["boot_diffs"]["response_vs_uniform"]
        assert np.allclose(lhs, boot["boot_diffs"]["uplift_vs_response"], atol=1e-12)

    def test_10_end_to_end_fixture_and_sql_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fix_path = tmp / "fixture.csv.gz"
            out_dir = tmp / "out"
            generate_synthetic_fixture(fix_path, n_rows=2000, seed=2026)

            record = run_pipeline(
                mode="fixture",
                data_path=fix_path,
                output_dir=out_dir,
                sample_size=2000,
                budget_share=0.20,
                regularization_c=1.0,
                bootstrap_replicates=50,
                sample_seed=11,
                split_seed=12,
                model_seed=13,
                bootstrap_seed=14,
            )
            assert record["heldout_headline_summary"]["sql_recomputation_all_agree"] is True
            sql_check = recompute_headline_metrics_sql(
                eval_csv_path=out_dir / "heldout_eval_rows.csv",
                sql_path=REPO_ROOT / "sql" / "verify_headline_metrics.sql",
                python_metrics={
                    "v_uplift_per_1k": record["heldout_headline_summary"]["primary_values_per_1k"]["uplift"],
                },
            )
            assert sql_check["all_metrics_agree"] is True

            # Verify resume renderer blocks synthetic fixture outputs
            with self.assertRaises(ValueError):
                render_bullets_from_real_run(out_dir)

    def test_11_real_mode_missing_file_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "criteo-research-uplift-v2.1.csv.gz"
            with self.assertRaises(FileNotFoundError):
                run_pipeline(
                    mode="real",
                    data_path=missing,
                    output_dir=Path(tmpdir) / "out",
                    sample_size=1000,
                    budget_share=0.20,
                    regularization_c=1.0,
                    bootstrap_replicates=10,
                    sample_seed=1,
                    split_seed=2,
                    model_seed=3,
                    bootstrap_seed=4,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
