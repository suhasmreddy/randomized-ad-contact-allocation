# Randomized Ad-Contact Allocation under a 20% Eligibility Budget

An offline causal policy-evaluation benchmark built on the **corrected Criteo Uplift Prediction Dataset (`criteo-research-uplift-v2.1.csv.gz`)** ([Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/); [Hugging Face Mirror](https://huggingface.co/datasets/criteo/criteo-uplift/tree/main)).

## Business Decision & Policies Compared

A campaign planner has budget to grant **ad eligibility (`treatment = 1`) to 20% of an eligible user cohort** ($K = \lfloor 0.20 \times N \rfloor$). Each eligibility slot consumes one unit of budget, and the objective is to maximize **incremental site visits (`visit = 1`)** relative to assigning no one.

Why might the user most likely to visit be a poor use of an ad slot? A high predicted probability of visiting under treatment, $P(Y=1 \mid X, T=1)$, conflates **baseline organic propensity** ("sure things" who visit even when held in control, $P(Y=1 \mid X, T=0) \approx P(Y=1 \mid X, T=1)$) with **causal ad lift**. Spending scarce eligibility slots on organic visitors crowds out lower-baseline "persuadable" users whose visit rate increases meaningfully only when eligible for advertising.

At identical budget $K = \lfloor 0.20 \times N \rfloor$ on the held-out test split, we evaluate three policies via Horvitz-Thompson inverse-propensity weighting (IPW):

1. **Uniform targeting (untargeted baseline):** Every user receives exact selection probability $\pi_i = K / N$ (evaluated directly in expectation).
2. **Response targeting (predictive-propensity baseline):** Select the $K$ users with the highest predicted treated visit probability $\hat{\mu}_1(X_i) = \hat{P}(Y=1 \mid X_i, T=1)$.
3. **Uplift targeting (causal CATE policy):** Select the $K$ users with the largest predicted uplift $\hat{\tau}(X_i) = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)$, sharing the exact same $\hat{\mu}_1(X_i)$ predictions as Response targeting.

## Reproduction Commands

```bash
# 1. Install pinned laptop dependencies and run the 11-check verification test suite
pip install -r requirements.txt
pytest -q

# 2. Small labeled synthetic fixture run (completes in ~3 seconds; verifies pipeline & SQL checks)
python3 -m src.cli --mode fixture

# 3. Full 200,000-row uniform Criteo v2.1 benchmark run (after placing archive at data/raw/)
#    Optional downloader (tries Criteo primary URL, then Hugging Face mirror, and checks SHA-256):
python3 scripts/fetch_criteo_v21.py
python3 -m src.cli --mode real --data-path data/raw/criteo-research-uplift-v2.1.csv.gz
```

## Results (real Criteo Uplift v2.1 run)

200,000-row uniform sample; held-out N = 40,000, K = 8,000 (20% budget).
Units: estimated incremental visits per 1,000 eligible users. Overall
treated-minus-control visit lift on the held-out set: +0.63 pp.

| Policy / comparison | Estimate | 95% paired-bootstrap CI |
|---|---:|---:|
| Uniform | +1.37 | n/a |
| Response targeting | +6.65 | n/a |
| Uplift targeting | +6.34 | n/a |
| **Uplift vs. response (primary)** | **-0.30** | **[-2.41, +1.86]** |
| Uplift vs. uniform | +4.97 | [+0.72, +8.77] |
| Response vs. uniform | +5.27 | [+0.69, +9.43] |
| Uplift vs. uniform (e(X) sensitivity) | +2.90 | [-1.53, +6.91] |

**Takeaways.** (1) The benchmark does not establish a winner between uplift
and response targeting. (2) Both targeted policies beat uniform under the
pooled-propensity assumption, but that advantage is not distinguishable from
zero under the covariate-propensity sensitivity check. (3) Treatment rates are
stable across score quintiles (all |z| < 1.3), and the SQL recomputation agrees
with Python (max abs diff 1.2e-13 across 7 headline metrics). The top uplift
quintile is also the highest-baseline group, which likely explains why response
targeting keeps pace here.

Files: `outputs/real/policy_results.csv`, `policy_differences.csv`,
`score_band_balance.csv`, `run_record.json`.

### Labeled Synthetic Fixture Verification Summary (`outputs/fixture/` — Pipeline Check Only, Not Empirical Criteo Data)

| Metric (`estimator = primary_pooled_e`) | Synthetic Fixture Value | 95% Paired Bootstrap CI (`1,000` draws) | Source Artifact |
| :--- | ---: | ---: | :--- |
| Held-out cohort $N$ / Quota $K$ (20%) | `4,000` / `800` | Fixed per draw ($K_b = \lfloor 0.20 N_b \rfloor$) | `outputs/fixture/policy_results.csv` |
| Uniform targeting $V(\pi^{\text{unif}})$ | `+1.09` visits / 1k | `[-3.12, +4.97]` | `outputs/fixture/policy_results.csv` |
| Response targeting $V(\pi^{\text{resp}})$ | `+8.99` visits / 1k | `[-6.39, +21.79]` | `outputs/fixture/policy_results.csv` |
| Uplift targeting $V(\pi^{\text{uplift}})$ | `+9.83` visits / 1k | `[-3.04, +22.27]` | `outputs/fixture/policy_results.csv` |
| **Primary:** $\Delta(\text{uplift}, \text{response})$ | `+0.84` visits / 1k | `[-8.51, +12.98]` (spans 0) | `outputs/fixture/policy_differences.csv` |
| **Secondary:** $\Delta(\text{uplift}, \text{uniform})$ | `+8.74` visits / 1k | `[-1.92, +18.96]` (spans 0) | `outputs/fixture/policy_differences.csv` |
| Independent SQL vs. Python max $\|\text{diff}\|$ | `2.31e-14` | Exact floating-point agreement | `outputs/fixture/sql_recomputation_check.json` |

## Methodology & Consequential Design Assumptions

1. **Streaming Uniform Sample (`200,000` rows, fixed prior to test evaluation):** `src/sampling.py::stream_uniform_sample` streams the full `13,979,592`-row gzipped archive in `250,000`-row chunks and assigns each row an independent `PCG64(seed=20261112)` priority key, keeping the `200,000` lowest-key rows. Every archive row has identical inclusion probability (`200,000 / 13,979,592 ≈ 1.43%`), avoiding head-of-file ordering bias while fitting comfortably on a laptop CPU. Sample size is fixed as a one-day computational constraint rather than tuned for statistical power.
2. **Intention-to-Treat (ITT) Scope & Strict Leakage Prevention:** `treatment` represents randomized ad eligibility, whereas `exposure` occurs *after* assignment (only a subset of eligible users are actually served an impression). Conditioning on or predicting with `exposure` introduces post-treatment selection bias. `src/schema.py::extract_feature_matrix` enforces a strict allowlist of verified pre-assignment features `f0`–`f11` and rejects `exposure`, `visit`, `conversion`, `treatment`, and identifiers.
3. **Frozen Two-Model T-Learner (`C = 1.0`) & Deterministic Tie-Breaking:** Preprocessing (`StandardScaler`) and two L2-regularized logistic regressions ($\hat{\mu}_1$ on treated training rows, $\hat{\mu}_0$ on control training rows; `seed=20261114`) are fit strictly on the `60%` training split (`split_seed=20261113`). Validation (`20%`) is used only for log-loss, Brier, and calibration diagnostics before opening the held-out `20%` test split (`N = 40,000`, $K = 8,000$ in real mode). Tied policy scores are broken deterministically by `(source_row_id, within_cohort_slot)` so ranked policies always select **exact quota $K$** even when duplicate rows appear in bootstrap draws.
4. **Horvitz-Thompson Policy Evaluator & Sensitivity Check:** For held-out user $i$,
   $$z_i = \frac{T_i Y_i}{e_i} - \frac{(1 - T_i) Y_i}{1 - e_i}, \qquad V(\pi) = \frac{1000}{N} \sum_{i=1}^{N} \pi_i z_i, \qquad \Delta(\pi, b) = V(\pi) - V(b)$$
   where the denominator $N$ is the entire eligible cohort (putting all policies on a common campaign-wide scale: incremental visits relative to no assignment per 1,000 eligible users). The primary estimator sets $e_i = \bar{T}_{\text{train}}$. Because Criteo assembled multiple incrementality tests and applied non-uniform privacy subsampling, true per-trial assignment probabilities are unobserved; we pair `score_band_balance.csv` with a covariate propensity sensitivity run $\hat{e}(X_i)$ (`estimator = sensitivity_propensity_e_x`). Note what these checks can and cannot establish: observed balance across `f0`–`f11` and score quintiles checks whether treatment rates vary across *measured* projections, but **cannot rule out unmeasured trial-level propensity differences** hidden by anonymization and subsampling.
5. **Paired Held-Out Bootstrap (`1,000` replicates):** `src/evaluation.py::paired_bootstrap_evaluation` (`bootstrap_seed=20261115`) resamples the $N$ held-out users with replacement using one shared index vector across all three policies per draw, keeping the fitted models fixed while recomputing $K_b$ and re-selecting the top-$K_b$ users inside each resampled cohort.

## Scope Limitations & Dataset Attribution

- **Pooled Trials & Privacy Subsampling:** As documented by [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), the dataset pools multiple advertiser incrementality tests and applies non-uniform privacy subsampling and random feature projection so true campaign lift levels and advertiser identities cannot be recovered. Empirical conclusions apply strictly to this public benchmark—not general ad-tech ROI or live dollar savings.
- **License & Citation:** `criteo-research-uplift-v2.1.csv.gz` is published by Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini (*"A Large Scale Benchmark for Uplift Modeling"*, AdKDD 2018) under **CC BY-NC-SA 4.0**. Raw data is excluded from version control (`.gitignore`); see `LICENSE_DATA_NOTICE.md` for full attribution and license terms.

## Repository Layout

- `src/schema.py` — Schema verification, `f0`–`f11` allowlist, and official Criteo v2.1 SHA-256 constants.
- `src/sampling.py` — Chunked priority-key uniform sampler, 60/20/20 split assignment, and synthetic fixture builder.
- `src/audit.py` & `sql/` — DuckDB SQL experiment audit (`sql/experiment_audit.sql`) and independent SQL headline recomputation (`sql/verify_headline_metrics.sql`), with automatic `sqlite3` stdlib fallback when running offline without `duckdb`.
- `src/modeling.py` — Train-only standardization, two-model L2 logistic T-learner, propensity model $\hat{e}(X)$, and validation calibration diagnostics (`scikit-learn` primary, exact `scipy` L-BFGS-B fallback).
- `src/policies.py` — Exact quota $K = \lfloor 0.20 N \rfloor$, uniform expectation, and deterministic top-$K$ tie-breaking.
- `src/evaluation.py` — Horvitz-Thompson IPW evaluator, 1,000-draw paired bootstrap, score-band balance, and $e(X)$ overlap/ESS diagnostics.
- `src/plotting.py` & `src/cli.py` — Two-panel interval chart with zero reference lines and single-command CLI runner.
- `scripts/` — `fetch_criteo_v21.py` (primary + Hugging Face fallback downloader) and `render_resume_bullets.py`.
- `docs/` — `decision_memo.md` (prospective pilot recommendation) and `resume_bullets.md` (metric lineage guide).
- `tests/test_allocation.py` — 11 automated tests (`pytest` and `unittest` compatible).
