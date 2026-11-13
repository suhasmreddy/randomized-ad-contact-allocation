-- Independent SQL Recomputation of Headline Policy Metrics from Exported Held-Out Rows
-- Verifies Python evaluator arithmetic:
--   z_i = T_i * Y_i / e_i - (1 - T_i) * Y_i / (1 - e_i)
--   V(pi) = (1000.0 / N) * SUM(pi_i * z_i)
--   Delta(pi, b) = V(pi) - V(b)

WITH recomputed_rows AS (
    SELECT
        source_row_id,
        treatment,
        visit,
        e_pooled,
        pi_uniform,
        pi_response,
        pi_uplift,
        (
            (CAST(treatment AS DOUBLE) * CAST(visit AS DOUBLE)) / e_pooled
            - ((1.0 - CAST(treatment AS DOUBLE)) * CAST(visit AS DOUBLE)) / (1.0 - e_pooled)
        ) AS z_sql
    FROM heldout_eval_rows
),
agg AS (
    SELECT
        COUNT(*) AS n_heldout,
        SUM(pi_response) AS quota_response,
        SUM(pi_uplift) AS quota_uplift,
        (1000.0 / COUNT(*)) * SUM(pi_uniform * z_sql) AS v_uniform_per_1k,
        (1000.0 / COUNT(*)) * SUM(pi_response * z_sql) AS v_response_per_1k,
        (1000.0 / COUNT(*)) * SUM(pi_uplift * z_sql) AS v_uplift_per_1k,
        100.0 * (
            AVG(CASE WHEN treatment = 1 THEN CAST(visit AS DOUBLE) END)
            - AVG(CASE WHEN treatment = 0 THEN CAST(visit AS DOUBLE) END)
        ) AS overall_itt_visit_diff_pp
    FROM recomputed_rows
)
SELECT
    n_heldout,
    quota_response,
    quota_uplift,
    v_uniform_per_1k,
    v_response_per_1k,
    v_uplift_per_1k,
    (v_uplift_per_1k - v_response_per_1k) AS delta_uplift_vs_response_per_1k,
    (v_uplift_per_1k - v_uniform_per_1k) AS delta_uplift_vs_uniform_per_1k,
    (v_response_per_1k - v_uniform_per_1k) AS delta_response_vs_uniform_per_1k,
    overall_itt_visit_diff_pp
FROM agg;
