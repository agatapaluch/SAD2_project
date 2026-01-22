#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Plot plateau effect: reconstruction accuracy vs total data volume.
run:
python2.7 scripts/plot_plateau.py \
  --group-csv biological/results/sweep_tool/group_eval.csv \
  --out biological/plots/plateau_jaccard.png

"""

from __future__ import print_function, unicode_literals

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def safe_tight_layout():
    # matplotlib+py2 sometimes fails on tight_layout due to text handling
    try:
        plt.tight_layout()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-csv", required=True, help="Path to group_eval.csv")
    parser.add_argument("--out", default="plateau_jaccard.png", help="Output plot path")
    parser.add_argument(
        "--metric",
        default="edge_jaccard_distance",
        choices=["edge_jaccard_distance", "graph_edit_distance"],
        help="Which metric to plot",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.group_csv)

    # Create total data volume (use ASCII '*' to avoid unicode issues in py2)
    df["data_volume"] = df["n_trajectories"] * df["trajectory_len"]

    # Aggregate mean metric for identical volumes
    agg = (
        df.groupby(["data_volume", "sync"])[args.metric]
        .mean()
        .reset_index()
        .sort_values("data_volume")
    )

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=agg, x="data_volume", y=args.metric, hue="sync", marker="o")

    plt.xlabel("Total data volume (n_trajectories * trajectory_len)")
    if args.metric == "edge_jaccard_distance":
        plt.ylabel("Edge Jaccard distance")
        plt.title("Plateau effect: Edge Jaccard vs data volume")
    else:
        plt.ylabel("Graph edit distance")
        plt.title("Plateau effect: GED vs data volume")

    safe_tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(args.out)
    plt.close()

    print("Saved plateau plot to:", args.out)


if __name__ == "__main__":
    main()
