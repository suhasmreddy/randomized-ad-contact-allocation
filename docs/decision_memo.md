# Decision Memo: Ad-Eligibility Allocation Policy for a Prospective 20% Budget Pilot

**To:** Campaign Planning & Marketing Science Review  
**From:** Decision Science Portfolio Analysis  
**Subject:** Prospective Pilot Recommendation — Uplift vs. Response vs. Uniform Ad-Eligibility Targeting  
**Benchmark Scope:** Offline evaluation pipeline on Criteo Uplift v2.1 (`200,000`-row uniform sample; `20%` eligibility quota; primary outcome: `visit`)

## 1. Empirical Status Note (No Synthetic Substitution)

Per strict analytical governance, **no synthetic fixture numbers are reported in this memo**. Because the build container operated without outbound internet access, the `311.4 MB` corrected Criteo archive (`criteo-research-uplift-v2.1.csv.gz`) must be run locally via one command (`python3 -m src.cli --mode real --data-path data/raw/criteo-research-uplift-v2.1.csv.gz`) to populate `outputs/real/policy_results.csv`, `outputs/real/policy_differences.csv`, `outputs/real/score_band_balance.csv`, and `outputs/real/run_record.json`. All estimator logic, leakage guards, SQL cross-checks, and 1,000-draw paired bootstrap routines have been verified against deterministic unit tests (`11/11` passing).

## 2. Structural Problem: Why High Visit Probability Can Waste Ad Slots

When ad eligibility is capped at $20\%$ of the audience ($K = \lfloor 0.20 N \rfloor$), **Response targeting** ranks users by $\hat{P}(\text{visit}=1 \mid X, T=1)$. That objective maximizes *observed visits among selected users*, not *incremental visits caused by eligibility*. In digital retargeting, user activity features (`f0`–`f11`) strongly predict organic return visits ("sure things" where $P(Y(0)=1 \mid X)$ is already high). Assigning a scarce ad slot to a user with a $50\%$ treated visit probability and a $49\%$ control visit probability yields only $+1\text{ pp}$ of incremental lift while consuming the same budget unit as a "persuadable" user whose visit probability rises from $2\%$ to $8\%$ ($+6\text{ pp}$ lift).

However, **Uplift targeting** ($\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$ via a two-model logistic T-learner) pays a real statistical price: subtracting two noisy probability estimates compounds estimation variance, especially because the Criteo benchmark assigns $\sim 85\%$ of users to treatment and only $\sim 15\%$ to control, making the control model $\hat{\mu}_0(X)$ ~5.7× smaller in sample size than $\hat{\mu}_1(X)$ and giving each control visitor a large negative Horvitz-Thompson weight ($-1/(1-e) \approx -6.7$).

## 3. Prospective Pilot Recommendation & Pre-Committed Decision Rule

Once `outputs/real/` is generated from the `200,000`-row Criteo v2.1 sample (`N = 40,000` held-out users, $K = 8,000$), apply the following pre-committed pilot recommendation rule without post-hoc threshold tuning:

1. **Recommend a Three-Arm Prospective Online Pilot (Uplift vs. Response vs. Small Uniform Holdout) with Uplift as Primary Challenger** *if* in `outputs/real/policy_differences.csv` (`estimator = primary_pooled_e`), `uplift_vs_response` has a positive point estimate ($\Delta > 0$), its 95% paired bootstrap confidence interval excludes zero (`ci95_low > 0`), **and** the policy ranking (`primary_ranking` vs. `sensitivity_ranking` in `outputs/real/run_record.json`) remains unchanged under the covariate propensity sensitivity check $\hat{e}(X)$.
2. **Treat the Offline Benchmark as Inconclusive and Recommend an Equal-Split Exploration Pilot (or Simpler Baseline)** *if* the 95% paired bootstrap interval for `uplift_vs_response` or `uplift_vs_uniform` spans zero (`conclusion = inconclusive_spans_zero` in `outputs/real/policy_differences.csv`). At a fixed one-day computational sample of `200,000` rows (`~6,000` held-out control users) and an overall Criteo v2.1 visit rate near $4.7\%$, wide Horvitz-Thompson intervals are common; an inconclusive interval means this `200,000`-row offline slice lacks precision to crown a single production winner, not that causal targeting fails in principle.
3. **Flag Propensity Instability Before Any Pilot** *if* `ranking_changed_under_e_x_sensitivity` is `true` in `outputs/real/run_record.json` or if `outputs/real/score_band_balance.csv` exhibits substantial treatment-rate divergence across uplift quintiles ($|z| > 2.5$). Because Criteo pooled multiple advertiser trials and applied non-uniform privacy subsampling, any ranking reversal between pooled $e$ and $\hat{e}(X)$ indicates sensitivity to cross-stratum assignment heterogeneity that offline anonymized features can only partially adjust for.

## 4. What Evidence Would Change My Mind?

Even if Uplift targeting wins cleanly on `outputs/real/`, three concrete findings would reverse a recommendation to deploy this two-model T-learner in a live campaign:

- **Cost and Exposure Asymmetry (ITT vs. Realized Ad Spend):** In this dataset, `treatment` is *eligibility*, while `exposure` occurs post-assignment. If Uplift targeting selects low-activity users whom the bidding auction rarely exposes—or whose impression CPM is substantially higher than Response-targeted users—incremental visits *per dollar spent* could invert relative to incremental visits *per eligibility slot*. A prospective pilot must log per-slot ad spend and impression delivery alongside ITT eligibility.
- **Downstream Conversion Divergence:** Optimizing `visit` can over-select curiosity clickers who never purchase. If re-evaluating the identical policy selections on `conversion` yields zero or negative incremental conversions (`Δ_conversion <= 0`), budget allocation should pivot to conversion-weighted or value-weighted uplift.
- **Calibration Drift in $\hat{\mu}_0(X)$:** If `outputs/real/validation_diagnostics.csv` shows poor control-arm calibration or higher validation log-loss than an intercept-only baseline, two-model differencing is injecting noise; regularized single-stage causal learners (e.g., R-learner / DR-learner trained on a larger sample with randomized holdout propensities logged directly at assignment time) should be tested before scaling budget.
