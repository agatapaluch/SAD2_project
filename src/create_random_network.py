# -*- coding: utf-8 -*-
import random
import os
import itertools


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def random_truth_table(num_parents, rng):
    """
    Return a random Boolean function as a truth table:
    dict mapping tuple of parent bits -> 0/1.
    """
    table = {}
    for bits in itertools.product([0, 1], repeat=num_parents):
        table[bits] = rng.randint(0, 1)
    return table


def random_boolean_network(n, max_parents, rng, allow_self_parent=False):
    """
    Construct one random Boolean network of size n with in-degree <= max_parents.

    Network format:
      {
        "n": n,
        "parents": [tuple_of_parent_indices_per_node],
        "truth_tables": [dict mapping parent_state_tuple -> 0/1 per node]
      }
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if max_parents < 0:
        raise ValueError("max_parents must be >= 0")

    all_indices = list(range(n))
    parents_list = []
    tables_list = []

    for i in range(n):
        if allow_self_parent:
            candidates = all_indices
        else:
            candidates = [j for j in all_indices if j != i]

        # choose k parents (can be 0), but no more than max_parents
        k_max = min(max_parents, len(candidates))
        k = rng.randint(0, k_max)

        if k > 0:
            parents = tuple(rng.sample(candidates, k))
        else:
            parents = tuple()

        table = random_truth_table(len(parents), rng)

        parents_list.append(parents)
        tables_list.append(table)

    return {"n": n, "parents": parents_list, "truth_tables": tables_list}


def step(network, state):
    """Synchronous update of all nodes."""
    n = network["n"]
    if len(state) != n:
        raise ValueError("state must have length %d" % n)

    next_state = [0] * n
    for i in range(n):
        parents = network["parents"][i]
        inp = tuple(state[p] for p in parents)
        next_state[i] = network["truth_tables"][i][inp]
    return next_state


def simulate(network, steps, rng, initial_state=None):
    """Simulate synchronous dynamics for 'steps' steps; return trajectory."""
    n = network["n"]
    if initial_state is None:
        initial_state = [rng.randint(0, 1) for _ in range(n)]

    traj = [initial_state]
    s = initial_state
    for _ in range(steps):
        s = step(network, s)
        traj.append(s)
    return traj

def step_async_one(network, state, node_idx):
    """
    Update exactly one node (node_idx) using the current state.
    Returns a NEW state list (does not modify input state in-place).
    """
    n = network["n"]
    s2 = list(state)
    parents = network["parents"][node_idx]
    inp = tuple(s2[p] for p in parents)
    s2[node_idx] = network["truth_tables"][node_idx][inp]
    return s2


def simulate_async(network, steps, rng, initial_state=None):
    """
    Asynchronous simulation with NO macro/micro steps:
    - Each step updates exactly ONE randomly chosen node.
    - Records every step (trajectory length = steps + 1).
    """
    n = network["n"]
    if initial_state is None:
        state = [rng.randint(0, 1) for _ in range(n)]
    else:
        if len(initial_state) != n:
            raise ValueError("initial_state must have length %d" % n)
        state = list(initial_state)

    traj = [list(state)]

    for _ in range(steps):
        i = rng.randint(0, n - 1)  # choose one node to update
        parents = network["parents"][i]
        inp = tuple(state[p] for p in parents)
        state = list(state)  # copy
        state[i] = network["truth_tables"][i][inp]
        traj.append(list(state))

    return traj


def construct_several_networks(min_n=5, max_n=16, max_parents=3, seed=0):
    """
    Build multiple networks for each size in [min_n, max_n].
    Returns: {n: [network, network, ...], ...}
    """
    rng = random.Random(seed)
    result = {}

    for n in range(min_n, max_n + 1):
        result[n] = random_boolean_network(n, max_parents, rng)

    return result


def save_trajectories_exact_input3_format(output_txt_path, trajectories):
    """
    Save exactly like input3.txt:
        <blank>  EXP0:0  EXP0:1  ... EXP0:(T-1)
        G0       ...
        G1       ...
    trajectories: list of lists; each inner list is a 0/1 sequence over time.
    """
    if not trajectories:
        raise ValueError("trajectories is empty")

    T = len(trajectories[0])
    for i, tr in enumerate(trajectories):
        if len(tr) != T:
            raise ValueError("trajectory %d length %d != %d" % (i, len(tr), T))

    f = open(output_txt_path, "w")
    try:
        header = [""] + ["EXP0:%d" % t for t in range(T)]
        f.write("\t".join(header) + "\n")

        for i, tr in enumerate(trajectories):
            row = ["G%d" % i] + [str(int(x)) for x in tr]
            f.write("\t".join(row) + "\n")
    finally:
        f.close()

if __name__ == "__main__":
    UPDATE_MODE = "sync"  # "sync" or "async"

    nets_by_size = construct_several_networks(min_n=5, max_n=16, max_parents=3, seed=123)


    # picking one of the networks created and generating trajectiories of length = steps
    network_size = 16
    net = nets_by_size[network_size]
    rng = random.Random(999)
    steps = 40

    if UPDATE_MODE == "sync":
        traj = simulate(net, steps=steps, rng=rng)
        filename = "output_sync.txt"
    elif UPDATE_MODE == "async":
        traj = simulate_async(net, steps=steps, rng=rng)
        filename = "output_async.txt"
    else:
        raise ValueError("UPDATE_MODE must be 'sync' or 'async'")

    # Convert state-over-time into per-node time series (G0..G(n-1))
    T = len(traj)
    node_time_series = []
    for node_idx in range(net["n"]):
        node_time_series.append([traj[t][node_idx] for t in range(T)])

    # Save into generated_trajectories/
    out_dir = "generated_trajectories"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)

    save_trajectories_exact_input3_format(out_path, node_time_series)
    print("Saved to:", out_path)
