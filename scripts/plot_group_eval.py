#!/usr/bin/env python2
"""
Plot aggregated evaluation results from group_eval.csv.

Expected input: CSV with columns including
  - sync, sampling_frequency, n_trajectories, trajectory_len, attractor_state_percentage
  - edge_jaccard_distance, graph_edit_distance
Optional columns such as scoring, total_seconds, ged_timed_out can be present.
"""

from __future__ import print_function

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def plot_heatmap(df, value, row, col, title, out_path):
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": value})
    plt.title(title)
    plt.ylabel(row)
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar(df, x, hue, value, title, out_path):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=x, hue=hue, y=value, estimator=pd.Series.mean, ci="sd")
    plt.title(title)
    plt.ylabel(value)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(df, x, y, size, hue, title, out_path):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, size=size, hue=hue, palette="viridis", sizes=(40, 200))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot grouped evaluation metrics.")
    parser.add_argument("--group-csv", required=True, help="Path to group_eval.csv")
    parser.add_argument("--out-dir", default="plots", help="Output directory for plots")
    parser.add_argument(
        "--score-label",
        default=None,
        help="Optional label to include in plot filenames (e.g., the scoring method).",
    )
    parser.add_argument(
        "--eval-csv",
        default=None,
        help="Optional eval.csv; used to rebuild grouping if group_csv lacks n_nodes.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    df = pd.read_csv(args.group_csv)

    # If n_nodes is missing (old group_eval), try to rebuild grouping from eval.csv.
    if "n_nodes" not in df.columns and args.eval_csv:
        if os.path.exists(args.eval_csv):
            eval_df = pd.read_csv(args.eval_csv)
            group_cols = [
                "sync",
                "sampling_frequency",
                "n_trajectories",
                "trajectory_len",
                "n_nodes",
                "attractor_state_percentage",
            ]
            metrics = ["edge_jaccard_distance", "graph_edit_distance"]
            existing_cols = [c for c in group_cols if c in eval_df.columns]
            existing_metrics = [m for m in metrics if m in eval_df.columns]
            if existing_cols and existing_metrics:
                df = eval_df.groupby(existing_cols)[existing_metrics].mean().reset_index()
                print("Rebuilt grouped data from {}".format(args.eval_csv))
        else:
            print("Warning: n_nodes missing and eval_csv not found; node-based plots will be skipped.")

    label = "_{}".format(args.score_label) if args.score_label else ""

    # Bar: Jaccard by number of trajectories, split by sync.
    out = os.path.join(args.out_dir, "bar_jaccard_ntraj{}.png".format(label))
    plot_bar(
        df,
        x="n_trajectories",
        hue="sync",
        value="edge_jaccard_distance",
        title="Edge Jaccard by #trajectories (group means)",
        out_path=out,
    )

    # Bar: Jaccard by sampling frequency, split by sync.
    out = os.path.join(args.out_dir, "bar_jaccard_sampling{}.png".format(label))
    plot_bar(
        df,
        x="sampling_frequency",
        hue="sync",
        value="edge_jaccard_distance",
        title="Edge Jaccard by sampling frequency (group means)",
        out_path=out,
    )

    # Bar: Jaccard by trajectory length, split by sync.
    out = os.path.join(args.out_dir, "bar_jaccard_trajlen{}.png".format(label))
    plot_bar(
        df,
        x="trajectory_len",
        hue="sync",
        value="edge_jaccard_distance",
        title="Edge Jaccard by trajectory length (group means)",
        out_path=out,
    )

    # Bar: GED by sampling frequency, split by sync.
    out = os.path.join(args.out_dir, "bar_ged_sampling{}.png".format(label))
    plot_bar(
        df,
        x="sampling_frequency",
        hue="sync",
        value="graph_edit_distance",
        title="GED by sampling frequency (group means)",
        out_path=out,
    )

    # Bar: GED by trajectory length, split by sync.
    out = os.path.join(args.out_dir, "bar_ged_trajlen{}.png".format(label))
    plot_bar(
        df,
        x="trajectory_len",
        hue="sync",
        value="graph_edit_distance",
        title="GED by trajectory length (group means)",
        out_path=out,
    )

    # Bar: Jaccard by number of nodes, split by sync.
    out = os.path.join(args.out_dir, "bar_jaccard_nodes{}.png".format(label))
    if "n_nodes" in df.columns:
        plot_bar(
            df,
            x="n_nodes",
            hue="sync",
            value="edge_jaccard_distance",
            title="Edge Jaccard by #nodes (group means)",
            out_path=out,
        )

    # Bar: GED by number of nodes, split by sync.
    out = os.path.join(args.out_dir, "bar_ged_nodes{}.png".format(label))
    if "n_nodes" in df.columns:
        plot_bar(
            df,
            x="n_nodes",
            hue="sync",
            value="graph_edit_distance",
            title="GED by #nodes (group means)",
            out_path=out,
        )

    print("Plots written to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
