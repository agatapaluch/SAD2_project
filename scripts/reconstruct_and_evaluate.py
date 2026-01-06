#!/usr/bin/env python2
# coding: utf-8
"""Run BNFinder on example datasets and evaluate reconstructed networks.

This script automates three steps:
1) Iterate over rows in the provided metadata.csv to find datasets and their
   ground-truth graphs.
2) Reconstruct a network with BNFinder (`bnf`) for each dataset (unless
   --skip-bnf is passed) and store the SIF output.
3) Compute two structure-based distances between the reconstructed graph and the
   ground-truth Boolean network:
     - Edge Jaccard distance (1 - Jaccard index) over directed edge sets; this
       captures how many edges are mismatched relative to the union.
     - Graph edit distance (GED) from NetworkX as a global topology measure.
"""

from __future__ import print_function

import argparse
import os
import subprocess
import time
import multiprocessing

import networkx as nx
import pandas as pd


def load_truth_graph(path):
    """Parse the ground-truth graph file (format used in example_datasets/graphs)."""
    edges = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("# RULES"):
                break
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def load_sif_graph(path):
    """Parse BNFinder SIF output; edge sign is kept as an attribute."""
    g = nx.DiGraph()
    if not os.path.exists(path):
        raise IOError("Reconstructed SIF not found: {}".format(path))
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            src, sign, tgt = parts[0], parts[1], parts[2]
            g.add_edge(src, tgt, sign=sign)
    return g


def edge_sets(graph):
    return set((u, v) for u, v in graph.edges())


def edge_stats(true_g, pred_g):
    """Compute edge-wise distance metrics."""
    true_edges = edge_sets(true_g)
    pred_edges = edge_sets(pred_g)
    intersection = true_edges & pred_edges
    union = true_edges | pred_edges

    return {
        "edge_jaccard_distance": 1 - (len(intersection) / float(len(union))) if union else 0.0,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
    }


def _ged_worker(true_g, pred_g, q):
    """Worker to compute GED; needed for hard timeouts."""
    try:
        ged = nx.graph_edit_distance(true_g, pred_g)
        try:
            iterator = iter(ged)
        except TypeError:
            q.put(float(ged))
            return
        best = None
        for cost in iterator:
            best = cost if best is None else min(best, cost)
        if best is None:
            best = float("inf")
        q.put(float(best))
    except Exception as exc:  # pragma: no cover
        q.put(exc)


def safe_graph_edit_distance(true_g, pred_g, timeout_seconds=None):
    """Graph edit distance with a hard timeout; falls back to edge diff if timed out."""
    start = time.time()
    fallback = len(edge_sets(true_g) ^ edge_sets(pred_g))
    if timeout_seconds is None or timeout_seconds <= 0:
        try:
            val = nx.graph_edit_distance(true_g, pred_g)
            try:
                return float(val), False
            except TypeError:
                pass
        except Exception:
            return float(fallback), True

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_ged_worker, args=(true_g, pred_g, q))
    p.start()
    p.join(timeout_seconds)
    if p.is_alive():
        p.terminate()
        p.join()
        return float(fallback), True
    if q.empty():
        return float(fallback), True
    result = q.get()
    if isinstance(result, Exception):
        return float(fallback), True
    return float(result), False


def run_bnfinder(bnf_cmd, dataset_path, output_path, scoring=None, extra_args=None):
    """Execute BNFinder to reconstruct a network from the dataset."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = [bnf_cmd, "-e", dataset_path, "-n", output_path, "-l", "3"]
    if scoring:
        cmd.extend(["-s", scoring])
    if extra_args:
        cmd.extend(extra_args)
    try:
        start = time.time()
        subprocess.check_call(cmd)
        return True, time.time() - start
    except subprocess.CalledProcessError as exc:
        print("BNFinder failed for {}: {}".format(dataset_path, exc))
        return False, None


def evaluate_dataset(row, args):
    t_total_start = time.time()
    dataset_path = row["dataset_filepath"]
    truth_graph_path = row["graph_filepath"]
    recon_path = os.path.join(args.output_dir, "{}_bnf.sif".format(os.path.splitext(os.path.basename(dataset_path))[0]))

    bnf_ok = True
    elapsed = None
    if not args.skip_bnf:
        bnf_ok, elapsed = run_bnfinder(
            args.bnf_bin,
            dataset_path,
            recon_path,
            args.score,
            args.bnf_extra,
        )

    t_load_truth_start = time.time()
    true_g = load_truth_graph(truth_graph_path)
    t_load_truth = time.time() - t_load_truth_start
    try:
        t_load_pred_start = time.time()
        pred_g = load_sif_graph(recon_path)
        t_load_pred = time.time() - t_load_pred_start
    except IOError as exc:
        return {
            "dataset_id": row["dataset_id"],
            "dataset": os.path.basename(dataset_path),
            "graph_id": row["graph_id"],
            "graph": os.path.basename(truth_graph_path),
            "error": str(exc),
        }
    if not bnf_ok:
        # BNFinder reported an error but produced a file; still evaluate.
        pass

    # Ensure both graphs have identical node sets so GED is fair.
    all_nodes = set(true_g.nodes()) | set(pred_g.nodes())
    true_g.add_nodes_from(all_nodes)
    pred_g.add_nodes_from(all_nodes)

    t_jaccard_start = time.time()
    stats = edge_stats(true_g, pred_g)
    t_jaccard = time.time() - t_jaccard_start

    t_ged_start = time.time()
    ged_value, ged_timed_out = safe_graph_edit_distance(true_g, pred_g, args.ged_timeout)
    t_ged = time.time() - t_ged_start
    stats["graph_edit_distance"] = ged_value

    stats.update(
        {
            "dataset_id": row["dataset_id"],
            "dataset": os.path.basename(dataset_path),
            "graph_id": row["graph_id"],
            "graph": os.path.basename(truth_graph_path),
            "n_nodes": row["n_nodes"],
            "n_trajectories": row["n_trajectories"],
            "trajectory_len": row["trajectory_len"],
            "sampling_frequency": row["sampling_frequency"],
            "sync": row["sync"],
            "allow_self_parent": row["allow_self_parent"],
            "attractor_state_percentage": row.get("attractor_state_percentage", None),
            "bnf_runtime_seconds": elapsed,
            "load_truth_seconds": t_load_truth,
            "load_pred_seconds": t_load_pred,
            "jaccard_seconds": t_jaccard,
            "ged_seconds": t_ged,
            "ged_timed_out": ged_timed_out,
            "total_seconds": time.time() - t_total_start,
        }
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct networks with BNFinder and score them against ground truth."
    )
    parser.add_argument(
        "--metadata",
        default="example_datasets/metadata.csv",
        help="CSV with dataset/graph mapping (default: example_datasets/metadata.csv)",
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="*",
        type=int,
        help="Optional list of dataset_id values to process; defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        default="reconstructed_networks",
        help="Where to store BNFinder SIF outputs.",
    )
    parser.add_argument("--bnf-bin", default="bnf", help="BNFinder executable name/path.")
    parser.add_argument(
        "-s",
        "--score",
        default=None,
        help="BNFinder scoring method passed via -s (e.g., BDE, MDL).",
    )
    parser.add_argument(
        "--bnf-extra",
        nargs="*",
        default=[],
        help="Additional flags forwarded to BNFinder (e.g., scoring options).",
    )
    parser.add_argument(
        "--ged-timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to search for graph edit distance per dataset (best-so-far is used if timed out).",
    )
    parser.add_argument(
        "--skip-bnf",
        action="store_true",
        help="Do not run BNFinder; reuse existing SIF files in --output-dir.",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save evaluation summary as CSV.",
    )
    parser.add_argument(
        "--save-group-csv",
        default=None,
        help="Optional path to save grouped evaluation summary (sync/sampling/n_traj/len).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.score and args.output_dir == "reconstructed_networks":
        args.output_dir = "results_{}".format(args.score)
    if args.save_csv is None:
        args.save_csv = os.path.join(args.output_dir, "eval.csv")
    if args.save_group_csv is None:
        args.save_group_csv = os.path.join(args.output_dir, "group_eval.csv")
    meta = pd.read_csv(args.metadata)
    if args.dataset_ids:
        meta = meta[meta["dataset_id"].isin(args.dataset_ids)]
    if meta.empty:
        raise SystemExit("No datasets to process based on provided filters.")

    results = []
    for _, row in meta.iterrows():
        stats = evaluate_dataset(row, args)
        results.append(stats)
        if "edge_jaccard_distance" in stats and "graph_edit_distance" in stats:
            parts = []
            if stats.get("bnf_runtime_seconds") is not None:
                parts.append("bnf={:.2f}s".format(stats["bnf_runtime_seconds"]))
            if stats.get("load_truth_seconds") is not None:
                parts.append("load_truth={:.2f}s".format(stats["load_truth_seconds"]))
            if stats.get("load_pred_seconds") is not None:
                parts.append("load_pred={:.2f}s".format(stats["load_pred_seconds"]))
            if stats.get("jaccard_seconds") is not None:
                parts.append("jaccard={:.2f}s".format(stats["jaccard_seconds"]))
            if stats.get("ged_seconds") is not None:
                parts.append("ged={:.2f}s".format(stats["ged_seconds"]))
            if stats.get("total_seconds") is not None:
                parts.append("total={:.2f}s".format(stats["total_seconds"]))
            if stats.get("ged_timed_out"):
                parts.append("ged_timed_out=True")
            timing = " | " + ", ".join(parts) if parts else ""
            print("[dataset {id}] Jaccard={jac:.3f} GED={ged:.3f}{timing}".format(
                id=stats["dataset_id"],
                jac=stats["edge_jaccard_distance"],
                ged=stats["graph_edit_distance"],
                timing=timing,
            ))
        else:
            print("[dataset {id}] skipped (error: {err})".format(
                id=stats.get("dataset_id", "?"),
                err=stats.get("error", "unknown"),
            ))

    df = pd.DataFrame(results)
    metrics = ["edge_jaccard_distance", "graph_edit_distance"]
    df_valid = df.dropna(subset=metrics, how="any")
    if args.save_csv:
        save_dir = os.path.dirname(args.save_csv)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(args.save_csv, index=False)

    # Group by dataset characteristics to compare settings.
    group_cols = [
        "sync",
        "sampling_frequency",
        "n_trajectories",
        "trajectory_len",
        "n_nodes",
        "attractor_state_percentage",
    ]
    group_metrics = ["edge_jaccard_distance", "graph_edit_distance"]
    grouped = df_valid.groupby(group_cols)[group_metrics].mean().reset_index()
    if args.save_group_csv:
        save_dir = os.path.dirname(args.save_group_csv)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        grouped.to_csv(args.save_group_csv, index=False)

    # Pretty print a short summary for quick inspection.
    print("\nMeasures used: edge Jaccard distance (edge overlap) and graph edit distance (minimum topology edits).")
    summary_cols = ["dataset_id", "edge_jaccard_distance", "graph_edit_distance"]
    print("\nTop reconstructions by lowest edge Jaccard distance:")
    if not df_valid.empty:
        print(df_valid.sort_values("edge_jaccard_distance", ascending=True).head(10)[summary_cols])
    else:
        print("No successful reconstructions to summarize.")
    print("\nGrouped performance (sync, sampling_frequency, n_trajectories, trajectory_len):")
    if not grouped.empty and "edge_jaccard_distance" in grouped.columns:
        print(grouped.sort_values("edge_jaccard_distance", ascending=True).head(10))
    else:
        print("No grouped results available.")


if __name__ == "__main__":
    main()
