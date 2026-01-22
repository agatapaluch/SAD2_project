Short readme describing scripts in current catalog in execution order.

generate_bn_trajectory_dataset.py
- generates a random Boolean Network - ground truth
- simulates its dynamic over time - saves observations how the states change state by step in time
- creates states trajectories (0/1 values) - list of states, one after another as time passes, 0 0 1 -> 0 1 1 -> 1 0 1 -> ...
- supports synchronous and asynchronous updates
- saves trajectories in BNfinder input format
- saves the true network structure for later comparison


generate_bn_trajectories_from_grid.py
- defines a grid parameters:
    - number of trajectories,
    - Trajectory length,
    - Synchronous/asynchronous updates
- runs previous script generate_bn_trajectory_dataset.py many times in loop over parameters
- creates seperate folder for each parameter setting


reconstruct_and_evaluate.py
- Loads generated trajectory datasets
- runs BNfinder on each dataset
- Loads the corresponding ground-truth network
- Compares  inferred edges with true edges
- Calculates evaluation metrics:
    - precision
    - recall
    - F1 score
- saves evaluation results to a .csv table


plot_group_eval.py
- Load .csv table adn groups results by experiment parameters
-plots and compares results of previous step




Dataflow:
Boolean Network (ground truth)
        |
Simulated dynamics
        |
Time-series trajectories
        |
BNfinder input files
        |
BNfinder reconstruction
        |
Comparison with ground truth
        |
Evaluation scores
        |
Plots and report
