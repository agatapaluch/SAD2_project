#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Generate multiple trajectory datasets for different parameter settings
and create metadata.csv compatible with scripts/reconstruct_and_evaluate.py.

Example:
python2.7 scripts/make_datasets_and_metadata.py \
  --bnet biological/data/tool.bnet \
  --out-datasets biological/results/generated_trajectories/sweep_tool \
  --out-graphs biological/results/graphs/sweep_tool \
  --out-metadata biological/metadata/metadata_tool_sweep.csv \
  --seeds 0 1 2 \
  --sync true false \
  --sampling 1 2 \
  --ntraj 5 10 15 20 25 30 \
  --tlen 50 100 150 200
"""

from __future__ import print_function

import argparse
import csv
import os
import subprocess

# columns needes in metadata.csv - input for reconstruct_and_evaluate.py script
HEADER = [
    "dataset_id",
    "dataset_filepath",
    "graph_id",
    "graph_filepath",
    "n_nodes",
    "n_trajectories",
    "trajectory_len",
    "sampling_frequency",
    "sync",
    "allow_self_parent",
    "attractor_state_percentage",
]


def mkdirp(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def count_nodes_in_bnet(bnet_path):
    n = 0
    with open(bnet_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("targets"):
                continue
            # each remaining line defines one target node
            n += 1
    return n


def read_attractor_percentage(dataset_path):
    """
    Expect file:
      <dataset_base>_attractors.txt
    with header:
      trajectory,attractor_state_percentage
    and at least one row like:
      EXP0,0.93
    We will take the mean over all rows (works for many trajectories).
    """
    base, _ = os.path.splitext(dataset_path)
    attr_path = base + "_attractors.txt"
    if not os.path.exists(attr_path):
        return ""

    vals = []
    with open(attr_path, "r") as fh:
        header = fh.readline()
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                vals.append(float(parts[1]))
            except Exception:
                pass
    if not vals:
        return ""
    return sum(vals) / float(len(vals))


def str_to_bool(s):
    s = s.strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def run_generator(
    gen_script,
    bnet,
    out_dataset,
    out_graph,
    sync,
    allow_self,
    seed,
    ntraj,
    tlen,
    sampling,
):
    cmd = [
        "python2.7",  # works with pipeline run in Docker
        gen_script,
        "-b",
        bnet,
        "-t",
        str(ntraj),
        "-l",
        str(tlen),
        "-f",
        str(sampling),
        "-r",
        str(seed),
        "-o",
        out_dataset,
        "-g",
        out_graph,
    ]
    if sync:
        cmd.append("-s")
    if allow_self:
        cmd.append("-p")

    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bnet", required=True, help="Path to .bnet model")
    ap.add_argument(
        "--gen-script",
        default="scripts/generate_bn_trajectory_dataset.py",
        help="Path to generate_bn_trajectory_dataset.py",
    )
    ap.add_argument(
        "--out-datasets", required=True, help="Output directory for datasets"
    )
    ap.add_argument(
        "--out-graphs", required=True, help="Output directory for groundtruth graphs"
    )
    ap.add_argument("--out-metadata", required=True, help="Output metadata CSV path")

    ap.add_argument(
        "--seeds", nargs="+", type=int, default=[0], help="Seeds, e.g. 0 1 2"
    )
    ap.add_argument(
        "--sync", nargs="+", default=["true"], help="true/false list, e.g. true false"
    )
    ap.add_argument(
        "--allow-self-parent", nargs="+", default=["false"], help="true/false list"
    )
    ap.add_argument(
        "--sampling", nargs="+", type=int, default=[1], help="Sampling frequency list"
    )
    ap.add_argument(
        "--ntraj", nargs="+", type=int, default=[20], help="Number of trajectories list"
    )
    ap.add_argument(
        "--tlen", nargs="+", type=int, default=[100], help="Trajectory length list"
    )

    ap.add_argument(
        "--graph-id", default=None, help="Graph ID label (default: bnet file stem)"
    )

    args = ap.parse_args()

    mkdirp(args.out_datasets)
    mkdirp(args.out_graphs)
    mkdirp(os.path.dirname(args.out_metadata))

    n_nodes = count_nodes_in_bnet(args.bnet)
    graph_id = args.graph_id
    if graph_id is None:
        graph_id = os.path.splitext(os.path.basename(args.bnet))[0]

    # Prepare parameter grid
    sync_vals = [str_to_bool(x) for x in args.sync]
    allow_vals = [str_to_bool(x) for x in args.allow_self_parent]

    rows = []
    dataset_id = 0

    for sync in sync_vals:
        for allow_self in allow_vals:
            for sampling in args.sampling:
                for ntraj in args.ntraj:
                    for tlen in args.tlen:
                        for seed in args.seeds:
                            tag = "sync{sync}_self{self}_f{f}_t{t}_l{l}_r{r}".format(
                                sync=int(sync),
                                self=int(allow_self),
                                f=sampling,
                                t=ntraj,
                                l=tlen,
                                r=seed,
                            )

                            dataset_path = os.path.join(
                                args.out_datasets, "{}.txt".format(tag)
                            )
                            graph_path = os.path.join(
                                args.out_graphs, "{}_groundtruth.txt".format(graph_id)
                            )

                            # generate (graph gets overwritten with same content; harmless)
                            run_generator(
                                args.gen_script,
                                args.bnet,
                                dataset_path,
                                graph_path,
                                sync,
                                allow_self,
                                seed,
                                ntraj,
                                tlen,
                                sampling,
                            )

                            attr_pct = read_attractor_percentage(dataset_path)

                            rows.append(
                                {
                                    "dataset_id": str(dataset_id),
                                    "dataset_filepath": dataset_path,
                                    "graph_id": graph_id,
                                    "graph_filepath": graph_path,
                                    "n_nodes": str(n_nodes),
                                    "n_trajectories": str(ntraj),
                                    "trajectory_len": str(tlen),
                                    "sampling_frequency": str(sampling),
                                    "sync": "True" if sync else "False",
                                    "allow_self_parent": (
                                        "True" if allow_self else "False"
                                    ),
                                    "attractor_state_percentage": (
                                        ""
                                        if attr_pct == ""
                                        else "{:.6f}".format(attr_pct)
                                    ),
                                }
                            )
                            dataset_id += 1

    # write metadata.csv
    with open(args.out_metadata, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Wrote metadata:", args.out_metadata)
    print("Datasets generated:", len(rows))


if __name__ == "__main__":
    main()
