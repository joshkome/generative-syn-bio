"""
Correlate Evo2 NLL scores with measured RBS translation rates (Kosuri dataset).

Reads:  data/processed/kosuri_contexts_labeled.csv
Writes: results/rbs_kosuri_scoring/rbs_kosuri_stats.csv
        results/rbs_kosuri_scoring/rbs_kosuri_scatter.png
        results/rbs_kosuri_scoring/rbs_kosuri_distributions.png

Usage:
    python scripts/analyze_rbs_correlation.py
    python scripts/analyze_rbs_correlation.py \
        --input  data/processed/kosuri_contexts_labeled.csv \
        --output results/rbs_kosuri_scoring
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["evo2_neg_log_likelihood", "mean_xlat"]).copy()
    df["evo2_log_prob"] = -df["evo2_neg_log_likelihood"]
    df["log10_xlat"]   = np.log10(df["mean_xlat"])
    return df


def run_stats(df: pd.DataFrame) -> pd.DataFrame:
    x, y, ly = df["evo2_log_prob"].values, df["mean_xlat"].values, df["log10_xlat"].values
    rows = [
        ("Pearson r",  "evo2_log_prob vs mean_xlat (raw)",      *pearsonr(x, y)),
        ("Pearson r",  "evo2_log_prob vs log10(mean_xlat)",     *pearsonr(x, ly)),
        ("Spearman rho", "evo2_log_prob vs mean_xlat",          *spearmanr(x, y)),
        ("Kendall tau",  "evo2_log_prob vs mean_xlat",          *kendalltau(x, y)),
    ]
    stats = pd.DataFrame(rows, columns=["test", "variables", "coefficient", "p_value"])
    stats["significant_p05"] = stats["p_value"] < 0.05
    return stats


def plot_scatter(df: pd.DataFrame, stats: pd.DataFrame, output_dir: str):
    n   = len(df)
    x   = df["evo2_log_prob"].values
    r   = stats.loc[stats["variables"].str.contains("raw"),        "coefficient"].iloc[0]
    p_r = stats.loc[stats["variables"].str.contains("raw"),        "p_value"].iloc[0]
    r_l = stats.loc[stats["variables"].str.contains("log10"),      "coefficient"].iloc[0]
    p_l = stats.loc[stats["variables"].str.contains("log10"),      "p_value"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Evo2 Log-Likelihood vs. Translation Rate - Kosuri RBS Library\n"
        f"trimmed_sequence scoring (20 bp upstream + RBS) | n={n}",
        fontsize=12,
    )

    panels = [
        (df["mean_xlat"].values,   "mean_xlat",          r,   p_r,  "steelblue"),
        (df["log10_xlat"].values,  "log10(mean_xlat)",  r_l, p_l, "mediumseagreen"),
    ]

    for ax, (yvals, ylabel, coeff, pval, color) in zip(axes, panels):
        ax.scatter(x, yvals, alpha=0.65, edgecolors="none", s=40, color=color)

        m, b = np.polyfit(x, yvals, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, m * xline + b, color="tomato", linewidth=1.5)

        for _, row in pd.concat([df.nlargest(2, "mean_xlat"), df.nsmallest(2, "mean_xlat")]).iterrows():
            yv = np.log10(row["mean_xlat"]) if "log" in ylabel else row["mean_xlat"]
            ax.annotate(row["rbs_name"], (row["evo2_log_prob"], yv),
                        fontsize=7, xytext=(5, 3), textcoords="offset points")

        sig = "  *" if pval < 0.05 else ""
        ax.set_xlabel("Evo2 log-likelihood (per token)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"r = {coeff:+.3f}   p = {pval:.4f}{sig}")

    plt.tight_layout()
    path = os.path.join(output_dir, "rbs_kosuri_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


SYNTHETIC_OUTLIERS = {"DeadRBS", "salis-1-21", "salis-2-3"}


def _draw_log_scatter(ax, df: pd.DataFrame, title: str):
    """Draw a single log-scale scatter panel onto ax. Returns computed stats."""
    from scipy.stats import pearsonr, spearmanr

    x = df["evo2_log_prob"].values
    y = df["log10_xlat"].values

    r,   p_r   = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)

    ax.scatter(x, y, alpha=0.65, edgecolors="none", s=45, color="mediumseagreen")

    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(xline, m * xline + b, color="tomato", linewidth=1.5)

    for _, row in pd.concat([df.nlargest(2, "mean_xlat"), df.nsmallest(2, "mean_xlat")]).drop_duplicates().iterrows():
        ax.annotate(row["rbs_name"], (row["evo2_log_prob"], np.log10(row["mean_xlat"])),
                    fontsize=8, xytext=(5, 3), textcoords="offset points")

    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:+.3f}  (p = {p_r:.2e})\nSpearman rho = {rho:+.3f}  (p = {p_rho:.2e})\nn = {len(df)}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    ax.set_xlabel("Evo2 log-likelihood (per token)")
    ax.set_ylabel("log10(mean_xlat)")
    ax.set_title(title)


def plot_log_scatter(df: pd.DataFrame, stats: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_log_scatter(ax, df, "Evo2 Log-Likelihood vs. Translation Rate\nKosuri RBS Library - all sequences")
    plt.tight_layout()
    path = os.path.join(output_dir, "rbs_kosuri_log_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_synthetic_comparison(df: pd.DataFrame, output_dir: str):
    df_filtered = df[~df["rbs_name"].isin(SYNTHETIC_OUTLIERS)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Effect of Synthetic Outliers on Evo2 vs. Translation Rate Correlation",
        fontsize=12,
    )

    _draw_log_scatter(axes[0], df,          f"All sequences (n={len(df)})")
    _draw_log_scatter(axes[1], df_filtered, f"Synthetic outliers removed (n={len(df_filtered)})\nexcluded: {', '.join(sorted(SYNTHETIC_OUTLIERS))}")

    plt.tight_layout()
    path = os.path.join(output_dir, "rbs_kosuri_synthetic_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_distributions(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["evo2_neg_log_likelihood"], bins=25, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Evo2 NLL (per token)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"NLL distribution\n"
        f"range: {df['evo2_neg_log_likelihood'].min():.3f} - {df['evo2_neg_log_likelihood'].max():.3f}"
    )

    axes[1].hist(df["log10_xlat"], bins=25, color="mediumseagreen", edgecolor="white")
    axes[1].set_xlabel("log10(mean_xlat)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Translation rate distribution")

    plt.tight_layout()
    path = os.path.join(output_dir, "rbs_kosuri_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default="data/processed/kosuri_contexts_labeled.csv")
    parser.add_argument("--output", default="results/rbs_kosuri_scoring")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    print(f"Loaded {len(df)} labeled sequences")
    print(f"NLL range:       {df['evo2_neg_log_likelihood'].min():.4f} – {df['evo2_neg_log_likelihood'].max():.4f}")
    print(f"mean_xlat range: {df['mean_xlat'].min():.0f} – {df['mean_xlat'].max():.0f}")
    print()

    stats = run_stats(df)
    for _, row in stats.iterrows():
        sig = "*" if row["significant_p05"] else " "
        print(f"{sig} {row['test']:14s}  {row['coefficient']:+.4f}   p={row['p_value']:.4f}   {row['variables']}")

    stats_path = os.path.join(args.output, "rbs_kosuri_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"\nSaved: {stats_path}")

    sns.set_theme(style="whitegrid", context="notebook")
    plot_scatter(df, stats, args.output)
    plot_log_scatter(df, stats, args.output)
    plot_synthetic_comparison(df, args.output)
    plot_distributions(df, args.output)


if __name__ == "__main__":
    main()
