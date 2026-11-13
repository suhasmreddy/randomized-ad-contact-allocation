-- SQL Experiment Audit for Criteo Uplift v2.1 Sample
-- Compatible with DuckDB (primary) and SQLite (offline fallback).
-- Audits split sizes, arm counts, ITT visit rates, post-assignment exposure rates,
-- and missingness across all 16 schema columns.

SELECT
    split,
    COUNT(*) AS n_rows,
    SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END) AS n_treated,
    SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END) AS n_control,
    AVG(CAST(treatment AS DOUBLE)) AS treatment_rate,
    AVG(CAST(visit AS DOUBLE)) AS visit_rate_overall,
    AVG(CASE WHEN treatment = 1 THEN CAST(visit AS DOUBLE) END) AS visit_rate_treated,
    AVG(CASE WHEN treatment = 0 THEN CAST(visit AS DOUBLE) END) AS visit_rate_control,
    100.0 * (
        AVG(CASE WHEN treatment = 1 THEN CAST(visit AS DOUBLE) END)
        - AVG(CASE WHEN treatment = 0 THEN CAST(visit AS DOUBLE) END)
    ) AS itt_visit_diff_pp,
    AVG(CASE WHEN treatment = 1 THEN CAST(exposure AS DOUBLE) END) AS exposure_rate_treated,
    AVG(CASE WHEN treatment = 0 THEN CAST(exposure AS DOUBLE) END) AS exposure_rate_control,
    SUM(
        CASE WHEN f0 IS NULL OR f1 IS NULL OR f2 IS NULL OR f3 IS NULL
               OR f4 IS NULL OR f5 IS NULL OR f6 IS NULL OR f7 IS NULL
               OR f8 IS NULL OR f9 IS NULL OR f10 IS NULL OR f11 IS NULL
               OR treatment IS NULL OR conversion IS NULL OR visit IS NULL OR exposure IS NULL
        THEN 1 ELSE 0 END
    ) AS rows_with_any_null
FROM sample_data
GROUP BY split

UNION ALL

SELECT
    'ALL' AS split,
    COUNT(*) AS n_rows,
    SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END) AS n_treated,
    SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END) AS n_control,
    AVG(CAST(treatment AS DOUBLE)) AS treatment_rate,
    AVG(CAST(visit AS DOUBLE)) AS visit_rate_overall,
    AVG(CASE WHEN treatment = 1 THEN CAST(visit AS DOUBLE) END) AS visit_rate_treated,
    AVG(CASE WHEN treatment = 0 THEN CAST(visit AS DOUBLE) END) AS visit_rate_control,
    100.0 * (
        AVG(CASE WHEN treatment = 1 THEN CAST(visit AS DOUBLE) END)
        - AVG(CASE WHEN treatment = 0 THEN CAST(visit AS DOUBLE) END)
    ) AS itt_visit_diff_pp,
    AVG(CASE WHEN treatment = 1 THEN CAST(exposure AS DOUBLE) END) AS exposure_rate_treated,
    AVG(CASE WHEN treatment = 0 THEN CAST(exposure AS DOUBLE) END) AS exposure_rate_control,
    SUM(
        CASE WHEN f0 IS NULL OR f1 IS NULL OR f2 IS NULL OR f3 IS NULL
               OR f4 IS NULL OR f5 IS NULL OR f6 IS NULL OR f7 IS NULL
               OR f8 IS NULL OR f9 IS NULL OR f10 IS NULL OR f11 IS NULL
               OR treatment IS NULL OR conversion IS NULL OR visit IS NULL OR exposure IS NULL
        THEN 1 ELSE 0 END
    ) AS rows_with_any_null
FROM sample_data;
