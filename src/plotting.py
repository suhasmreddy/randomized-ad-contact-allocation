"""Single publication-style chart comparing policy values and paired bootstrap differences."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_policy_comparison(
    results_df: pd.DataFrame,
    diffs_df: pd.DataFrame,
    output_path: Path,
    dataset_mode: str,
) -> None:
    """Render a clean two-panel horizontal interval plot with explicit zero reference lines.

    Left panel: Estimated incremental visits per 1,000 eligible users V(pi) at 20% budget.
    Right panel: Paired bootstrap policy differences Delta(pi, baseline) with 95% CIs.
    """
    primary_res = results_df[results_df["estimator"] == "primary_pooled_e"].copy()
    primary_diff = diffs_df[diffs_df["estimator"] == "primary_pooled_e"].copy()

    policy_order = ["uniform", "response", "uplift"]
    policy_labels = {
        "uniform": "Uniform Targeting\n(π_i = K/N)",
        "response": "Response Targeting\n(Top-K P(Y=1 | X, T=1))",
        "uplift": "Uplift Targeting\n(Top-K [P_1(X) − P_0(X)])",
    }
    primary_res["_ord"] = primary_res["policy"].map({p: i for i, p in enumerate(policy_order)})
    primary_res = primary_res.sort_values("_ord").reset_index(drop=True)

    diff_order = ["response_vs_uniform", "uplift_vs_uniform", "uplift_vs_response"]
    diff_labels = {
        "response_vs_uniform": "Response − Uniform",
        "uplift_vs_uniform": "Uplift − Uniform\n(Secondary)",
        "uplift_vs_response": "Uplift − Response\n(Primary Comparison)",
    }
    primary_diff["_ord"] = primary_diff["comparison"].map({d: i for i, d in enumerate(diff_order)})
    primary_diff = primary_diff.sort_values("_ord").reset_index(drop=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    is_fixture = dataset_mode != "real_criteo_v2.1"
    banner_note = (
        "SYNTHETIC FIXTURE VERIFICATION RUN (NOT REAL CRITEO DATA)"
        if is_fixture
        else "Criteo Uplift v2.1 Held-Out Test Cohort (N = 40,000; 20% Ad-Eligibility Budget)"
    )
    title_color = "#9A3412" if is_fixture else "#0F172A"

    fig.suptitle(
        "Randomized Ad-Contact Allocation at 20% Eligibility Budget",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        banner_note,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold" if is_fixture else "normal",
        color=title_color,
    )

    label_bbox = dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="none", alpha=0.92)

    # Panel A: Policy values V(pi)
    y_pos1 = np.arange(len(primary_res))
    vals1 = primary_res["v_per_1000_eligible"].to_numpy()
    lo1 = primary_res["ci95_low"].to_numpy()
    hi1 = primary_res["ci95_high"].to_numpy()
    colors1 = ["#64748B", "#D97706", "#1D4ED8"]

    ax1.axvline(0.0, color="#334155", linestyle="--", linewidth=1.2, zorder=1, label="Zero incremental visits")
    for i in range(len(primary_res)):
        ax1.errorbar(
            vals1[i],
            y_pos1[i],
            xerr=[[vals1[i] - lo1[i]], [hi1[i] - vals1[i]]],
            fmt="o",
            color=colors1[i],
            ecolor=colors1[i],
            elinewidth=2.2,
            capsize=4.5,
            markersize=7.5,
            zorder=3,
        )
        ax1.annotate(
            f"{vals1[i]:+.2f} [{lo1[i]:+.2f}, {hi1[i]:+.2f}]",
            xy=(vals1[i], y_pos1[i]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=9.0,
            color="#0F172A",
            fontweight="bold" if primary_res.loc[i, "policy"] == "uplift" else "normal",
            bbox=label_bbox,
            zorder=4,
        )

    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels([policy_labels[p] for p in primary_res["policy"]])
    ax1.set_ylim(-0.55, len(primary_res) - 0.22)
    span1 = max(float(np.max(hi1) - np.min(lo1)), 1.0)
    ax1.set_xlim(min(float(np.min(lo1)), 0.0) - 0.18 * span1, max(float(np.max(hi1)), 0.0) + 0.18 * span1)
    ax1.set_xlabel("Incremental Visits per 1,000 Eligible Users (V)")
    ax1.set_title("A. Policy Value Relative to No Assignment")
    ax1.grid(axis="x", linestyle=":", alpha=0.55, color="#CBD5E1")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Paired Bootstrap Policy Differences Delta(pi, b)
    y_pos2 = np.arange(len(primary_diff))
    vals2 = primary_diff["delta_per_1000_eligible"].to_numpy()
    lo2 = primary_diff["ci95_low"].to_numpy()
    hi2 = primary_diff["ci95_high"].to_numpy()
    colors2 = ["#64748B", "#0D9488", "#1D4ED8"]

    ax2.axvline(0.0, color="#DC2626", linestyle="--", linewidth=1.4, zorder=1, label="No difference (Δ = 0)")
    for i in range(len(primary_diff)):
        is_primary = primary_diff.loc[i, "comparison"] == "uplift_vs_response"
        ax2.errorbar(
            vals2[i],
            y_pos2[i],
            xerr=[[vals2[i] - lo2[i]], [hi2[i] - vals2[i]]],
            fmt="D" if is_primary else "o",
            color=colors2[i],
            ecolor=colors2[i],
            elinewidth=2.5 if is_primary else 2.0,
            capsize=4.5,
            markersize=7.5,
            zorder=3,
        )
        ax2.annotate(
            f"{vals2[i]:+.2f} [{lo2[i]:+.2f}, {hi2[i]:+.2f}]",
            xy=(vals2[i], y_pos2[i]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=9.0,
            color="#0F172A",
            fontweight="bold" if is_primary else "normal",
            bbox=label_bbox,
            zorder=4,
        )

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([diff_labels[c] for c in primary_diff["comparison"]])
    ax2.set_ylim(-0.55, len(primary_diff) - 0.22)
    span2 = max(float(np.max(hi2) - np.min(lo2)), 1.0)
    ax2.set_xlim(min(float(np.min(lo2)), 0.0) - 0.18 * span2, max(float(np.max(hi2)), 0.0) + 0.18 * span2)
    ax2.set_xlabel("Incremental Visit Gain per 1,000 Eligible Users (Δ)")
    ax2.set_title("B. Paired Bootstrap Policy Differences (95% CI)")
    ax2.grid(axis="x", linestyle=":", alpha=0.55, color="#CBD5E1")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    n_boot = int(primary_diff["bootstrap_replicates"].iloc[0])
    fig.text(
        0.5,
        0.02,
        f"Horvitz-Thompson IPW estimator with pooled training treatment rate e; 95% CIs from {n_boot:,} paired held-out bootstrap draws.",
        ha="center",
        fontsize=8.8,
        color="#475569",
    )

    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
