# Resume Bullet Options & Exact Metric Lineage

> **Governance Rule (Zero Synthetic Numbers):** The three resume bullets below describe an offline analysis of public randomized-trial data (`criteo-research-uplift-v2.1.csv.gz`). Per project integrity rules, **synthetic fixture numbers are never inserted here**. Fixed experimental design constants (`200,000` sampled rows, `40,000` held-out users, `8,000` quota slots at a `20%` budget, `12` pre-assignment features, `1,000` paired bootstrap draws, `11` automated tests) are shown alongside explicit `<REAL_*>` tokens for empirical Criteo v2.1 outputs.
>
> Once you run `python3 -m src.cli --mode real --data-path data/raw/criteo-research-uplift-v2.1.csv.gz` locally, run:
> ```bash
> python3 scripts/render_resume_bullets.py --run-dir outputs/real
> ```
> to populate every `<REAL_*>` token directly from `outputs/real/` (the script enforces `dataset_mode == "real_criteo_v2.1"` and rejects `outputs/fixture/`).

## Three Resume Bullet Options (Data & Decision Science / Industrial Engineering)

1. **Option 1 — Decision & Causal Policy Evaluation Focus**  
   Evaluated constrained `20%` ad-eligibility allocation (`8,000` of `40,000` held-out users from a `200,000`-row uniform sample of the public Criteo Uplift v2.1 randomized-trial benchmark), estimating `<REAL_V_UPLIFT>` vs. `<REAL_V_RESPONSE>` incremental visits per `1,000` eligible users for two-model logistic uplift vs. response targeting (paired $\Delta =$ `<REAL_DELTA_UR>`, `95%` CI `[<REAL_CI_UR_LOW>, <REAL_CI_UR_HIGH>]` across `1,000` bootstrap draws).

2. **Option 2 — Econometric Diagnostics & Baseline Comparison Focus**  
   Built an offline Horvitz-Thompson inverse-propensity policy evaluator and DuckDB SQL experiment audit over `200,000` public randomized-trial records (`12` pre-assignment features; overall held-out ITT visit lift `<REAL_ITT_PP>` pp), benchmarking uplift targeting against a `<REAL_V_UNIFORM>` visits/`1,000` uniform baseline (paired $\Delta =$ `<REAL_DELTA_UU>`, `95%` CI `[<REAL_CI_UU_LOW>, <REAL_CI_UU_HIGH>]`) and testing ranking stability under covariate propensity $e(X)$ weights.

3. **Option 3 — Reproducible Engineering & Statistical Verification Focus**  
   Engineered a reproducible Python and DuckDB SQL causal policy-evaluation pipeline for public randomized-trial ad allocation (`200,000` streamed rows, `60/20/20` splits, `1,000` paired bootstrap replicates), verified by `11` automated `pytest` checks covering `4`-user hand-calculated IPW expectations, post-assignment leakage guards, deterministic tie-breaking, and SQL metric recomputation agreeing within `< 1e-12`.

## Complete Number-to-Code Lineage Table

Every constant and empirical token in the three bullets above traces directly to an output file, filter row, column name, mathematical definition, and source function:

| Resume Token / Constant | Output File (`outputs/real/`) | Row Filter | Column / JSON Key | Exact Calculation | Computing Function |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `200,000` sampled rows | `run_record.json` | — | `sampling_and_splits.sampled_rows` | Priority-key uniform stream sample over all `13,979,592` archive rows | `src/sampling.py::stream_uniform_sample` |
| `40,000` held-out users ($N$) | `policy_results.csv` | `estimator == "primary_pooled_e"` | `n_heldout_eligible` | `round(0.20 * sampled_rows)` held-out `test` split count | `src/sampling.py::assign_splits` |
| `8,000` quota slots ($K$) | `policy_results.csv` | `estimator == "primary_pooled_e"` | `quota_k` | `floor(0.20 * N)` at `budget_share = 0.20` | `src/policies.py::compute_budget_quota` |
| `12` pre-assignment features | `run_record.json` | — | `model_and_budget_settings.allowed_features` | `len(("f0", ..., "f11"))`; excludes `exposure`, `visit`, `conversion`, `treatment` | `src/schema.py::extract_feature_matrix` |
| `1,000` bootstrap draws | `policy_differences.csv` | `estimator == "primary_pooled_e"` | `bootstrap_replicates` | Shared-index held-out resamples with fixed models & recomputed $K_b$ | `src/evaluation.py::paired_bootstrap_evaluation` |
| `11` automated `pytest` checks | `tests/test_allocation.py` | — | `TestAllocationSuite` (`test_01`–`test_11`) | Unit + integration test count verified by `pytest -q` / `unittest` | `tests/test_allocation.py` |
| `< 1e-12` SQL agreement | `sql_recomputation_check.json` | — | `comparisons.*.abs_diff` | $\max \|V_{\text{Python}} - V_{\text{SQL}}\|$ across all 7 headline metrics | `src/audit.py::recompute_headline_metrics_sql` |
| `<REAL_ITT_PP>` | `run_record.json` | — | `heldout_headline_summary.overall_itt_visit_diff_pp` | $100 \times (\bar{Y}_{T=1,\text{test}} - \bar{Y}_{T=0,\text{test}})$ | `src/evaluation.py::compute_overall_itt_visit_diff_pp` |
| `<REAL_V_UNIFORM>` | `policy_results.csv` | `estimator == "primary_pooled_e" & policy == "uniform"` | `v_per_1000_eligible` | $(1000 / N) \sum_{i=1}^N (K/N) z_i(\bar{T}_{\text{train}})$ | `src/evaluation.py::evaluate_three_policies` |
| `<REAL_V_RESPONSE>` | `policy_results.csv` | `estimator == "primary_pooled_e" & policy == "response"` | `v_per_1000_eligible` | $(1000 / N) \sum_{i=1}^N \pi_i^{\text{resp}} z_i(\bar{T}_{\text{train}})$ | `src/evaluation.py::evaluate_three_policies` |
| `<REAL_V_UPLIFT>` | `policy_results.csv` | `estimator == "primary_pooled_e" & policy == "uplift"` | `v_per_1000_eligible` | $(1000 / N) \sum_{i=1}^N \pi_i^{\text{uplift}} z_i(\bar{T}_{\text{train}})$ | `src/evaluation.py::evaluate_three_policies` |
| `<REAL_DELTA_UR>` | `policy_differences.csv` | `estimator == "primary_pooled_e" & comparison == "uplift_vs_response"` | `delta_per_1000_eligible` | $V(\pi^{\text{uplift}}) - V(\pi^{\text{resp}})$ | `src/evaluation.py::evaluate_three_policies` |
| `<REAL_CI_UR_LOW>`, `<REAL_CI_UR_HIGH>` | `policy_differences.csv` | `estimator == "primary_pooled_e" & comparison == "uplift_vs_response"` | `ci95_low`, `ci95_high` | `2.5th` and `97.5th` percentiles of paired bootstrap differences | `src/evaluation.py::paired_bootstrap_evaluation` |
| `<REAL_DELTA_UU>` | `policy_differences.csv` | `estimator == "primary_pooled_e" & comparison == "uplift_vs_uniform"` | `delta_per_1000_eligible` | $V(\pi^{\text{uplift}}) - V(\pi^{\text{unif}})$ | `src/evaluation.py::evaluate_three_policies` |
| `<REAL_CI_UU_LOW>`, `<REAL_CI_UU_HIGH>` | `policy_differences.csv` | `estimator == "primary_pooled_e" & comparison == "uplift_vs_uniform"` | `ci95_low`, `ci95_high` | `2.5th` and `97.5th` percentiles of paired bootstrap differences | `src/evaluation.py::paired_bootstrap_evaluation` |
