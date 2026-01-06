import os
import shutil
import itertools
import pandas as pd

from tqdm import tqdm
from generate_bn_trajectory_dataset import BN


def reset_output_directory(output_dir):
    """Reset a directory for writing a new batch of datasets"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Restart dataset and graph directories
    dataset_dir = os.path.join(output_dir, "datasets")
    graph_dir = os.path.join(output_dir, "graphs")
    shutil.rmtree(dataset_dir, ignore_errors=True)
    shutil.rmtree(graph_dir, ignore_errors=True)
    os.makedirs(dataset_dir)
    os.makedirs(graph_dir)

    # Reset metadata file
    metadata_file = os.path.join(output_dir, "metadata.csv")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    return dataset_dir, graph_dir, metadata_file


def main(
        param_grid,
        output_dir,
):
    """Generate multiple BN trajectory datasets using a parameter grid."""
    # Create a clean output directory for the results if it does not exist
    dataset_dir, graph_dir, metadata_file = reset_output_directory(output_dir)

    # Boolean network generation parameters
    network_args = ["n_nodes", "sync", "allow_self_parent", "network_seed"]
    network_params = list(itertools.product(*[param_grid[arg] for arg in network_args]))
    n_networks = len(network_params)

    # Trajectory dataset generation parameters
    trajectory_args = ["n_trajectories", "trajectory_len", "sampling_frequency", "trajectory_seed"]
    trajectory_params = list(itertools.product(*[param_grid[arg] for arg in trajectory_args]))
    d = 0   # Dataset id
    metadata_rows = []

    # Generate random boolean networks
    for i, (n_nodes, sync, allow_self_parent, network_seed) in enumerate(network_params, start=1):
        print("\nInitializing Boolean Network {}/{}: n_nodes={}, sync={}, allow_self_parent={}, seed={}".format(i, n_networks, n_nodes, sync, allow_self_parent, network_seed))
        bn = BN(n_nodes=n_nodes, sync=sync, allow_self_parent=allow_self_parent, seed=network_seed)
        graph_name = "G" + str(i)
        graph_filepath = os.path.join(graph_dir, graph_name+".txt")
        bn.save_graph_structure(graph_filepath)

        # For each network generate a set of trajectory datasets
        for n_trajectories, trajectory_len, sampling_frequency, trajectory_seed in tqdm(trajectory_params, desc="Generating trajectory datasets"):
            dataset_name = "D" + str(d)
            dataset_filepath = os.path.join(dataset_dir, dataset_name+".txt")
            attractor_percentage = bn.generate_trajectory_dataset(dataset_filepath, n_trajectories, trajectory_len, sampling_frequency, random_seed=trajectory_seed, return_attractor_percentage=True, save_attractor_percengage=False)

            # Create dataset metadata row
            metadata_row = {
                "graph_id": i,
                "graph_filepath": graph_filepath,
                "dataset_id": d,
                "dataset_filepath": dataset_filepath,
                "n_nodes": n_nodes,
                "sync": sync,
                "allow_self_parent": allow_self_parent,
                "network_seed": network_seed,
                "n_trajectories": n_trajectories,
                "trajectory_len": trajectory_len,
                "sampling_frequency": sampling_frequency,
                "trajectory_seed": trajectory_seed,
                "attractor_state_percentage": attractor_percentage,
            }

            metadata_rows.append(metadata_row)
            
            # Increase dataset id
            d += 1

    # Save metadata to csv
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_file)



if __name__ == "__main__":

    # Construct a parameter grid
    param_grid = {
        # Network params
        "n_nodes": [5,8,11,16],   # Using large networks (n_nodes > 12) can result in a few minute wait (exponential complexity), so make a coffee
        "sync": [True, False],
        "allow_self_parent": [False],  # BNFinder does not allow self loops by default (can be turned on but most biological models don't have them either)
        "network_seed": [0,1],
        # Trajectory params
        "n_trajectories": [1, 5, 20],
        "trajectory_len": [10, 20, 100],
        "sampling_frequency": [1,3],
        "trajectory_seed": [0],
    }

    OUTPUT_DIR = "example_datasets/"

    main(param_grid, OUTPUT_DIR)