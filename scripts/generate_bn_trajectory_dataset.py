import os
import argparse
import copy
import boolean
import random
import networkx as nx


def generate_random_walks(graph, n_walks, walk_length, seed=None):
    """
    Generate random walks on a graph.

    Parameters
    ----------
    graph : graph-like
        graph[node] returns iterable of neighbors
    n_walks : int
        Number of random walks to generate
    walk_length : int
        Number of steps per walk (nodes = walk_length + 1)
    seed : int or None
        Random seed

    Returns
    -------
    walks : list[list]
        List of random walks (each is a list of visited nodes)
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes()) if hasattr(graph, "nodes") else list(graph.keys())

    if not nodes:
        raise ValueError("Graph has no nodes")

    walks = []

    for _ in range(n_walks):
        current = rng.choice(nodes)
        walk = [current]

        for _ in range(walk_length):
            neighbors = list(graph[current])

            if not neighbors:
                break  # dead end

            current = rng.choice(neighbors)
            walk.append(current)

        walks.append(walk)

    return walks


class BN():

    __bool_algebra = boolean.BooleanAlgebra()

    def __int_to_state(self, x):
        """
        Helper method for converting a non-negative integer into a state in the form of a tuple of 0s and 1s.
    
            Args:
                x (int): A state number
    
            Returns:
                tuple[int, ...]: A tuple of 0s and 1s representing the Boolean network state.
        """
        binary_str = format(x,'0'+str(self.num_nodes)+'b')
        state = [int(char) for char in binary_str]

        return tuple(state)


    @staticmethod
    def __state_to_binary_str(state):
        """
        Converts a Boolean network state from a tuple of 0s and 1s into a binary string.
    
            Args:
                state (tuple[int, ...]): A tuple of 0s and 1s representing the Boolean network state
    
            Returns:
                str: A binary string representing the Boolean network state
        """
        bin_str = ''
        for bit in state:
            bin_str += str(bit)
        
        return bin_str
    

    def generate_random_boolean_functions(self, list_of_nodes, max_parents, allow_self_parent = False):
        """
        Generates random boolean functions in boolean.py format from a list of nodes (max 3 parents per node). 
        """
        def _random_boolean_function(parent_nodes):
            """Create radnom boolean function with !, &, | operators."""
            symbol_list = []

            for i, node in enumerate(parent_nodes):
                # Random negation
                neg = self.rng.sample(["","!"],1)[0]
                variable = neg + node

                symbol_list.append(variable)

                if i == len(parent_nodes) - 1:
                    break

                # Random operator
                op = self.rng.sample(["&","|"],1)[0]
                symbol_list.append(" " + op + " ")

            return "".join(symbol_list)
        
        list_of_functions = []

        # For each node generate its governing boolean function
        for i, node in enumerate(list_of_nodes):
            list_of_candidates = copy.copy(list_of_nodes)

            # Drop node if no self parents allowed
            if not allow_self_parent:
                list_of_candidates.pop(i)

            # Max num node parents
            k_max = min(max_parents, len(list_of_candidates))

            # Choose number of parents
            k = self.rng.randint(0, k_max)

            if k > 0:
                parents = self.rng.sample(list_of_candidates, k)
                func = _random_boolean_function(parents)
            else:
                func = "TRUE"

            list_of_functions.append(func)

        return list_of_functions
    

    def generate_graph_structure(self):
        """Generate directed graph structure of the boolean network in nx.DiGraph format."""
        G = nx.DiGraph()

        nodes = [str(node) for node in self.list_of_nodes]
        edges = []

        for node, fun in zip(nodes, self.functions):
            for parent in fun.symbols:
                edges.append((str(parent), node))

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        return G
                

    def __init__(self,
                 n_nodes = None,
                 bnet_path = None,
                 list_of_nodes = None,
                 list_of_functions = None,
                 sync = False,
                 allow_self_parent = False,
                 seed = 42,
        ):
        """
        Class constructor
    
            Args:
                OPTION 1:
                n_nodes (int): number of nodes to generate a random boolean network from.

                OPTION 2:
                bnet_path (str): A .bnet format boolean network file.

                OPTION 3:
                list_of_nodes (list[str]): A list of node names
                list_of_functions (list[str]): A list of strings representing the Boolean functions for the corresponding nodes
                    in the list_of_nodes, e.g. '(x0 & ~x1) | x2', where 'x0', 'x1', and 'x2' are node names.

                OTHER ARGS:
                sync (bool): Boolean of whether to use synchronous (True) or asynchronous (False) state transition system
                allow_self_parent (bool): True if allow for a node to be its own parent (regulator) (default False)
                seed (int): random seed for boolean function generation (if list of functions is not provided)
             
        """
        # Set random seed
        self.rng = random.Random(seed)

        # Load BN
        if list_of_nodes is not None and list_of_functions is not None:
            pass
        elif n_nodes:
            list_of_nodes = ["x"+str(i) for i in range(n_nodes)]
            list_of_functions = self.generate_random_boolean_functions(list_of_nodes, max_parents=3, allow_self_parent=allow_self_parent)
        elif bnet_path:
            list_of_nodes, list_of_functions = self.parse_bnet(bnet_path)
        else:
            raise Exception("Problem loading BN: no node or function list provided. Specify either n_nodes, .bnet filepath or list of nodes and functions.")

        self.num_nodes = len(list_of_nodes)

        self.node_names = list_of_nodes

        self.list_of_nodes = []
        for node_name in list_of_nodes:
            node = self.__bool_algebra.Symbol(node_name)
            self.list_of_nodes.append(node)
        
        self.functions = []
        for fun in list_of_functions:
            self.functions.append(self.__bool_algebra.parse(fun,simplify=True))

        self.graph_structure = self.generate_graph_structure()
        self.sync = sync
        self.sts = self.generate_state_transition_system()
        self.attractors = self.get_attractors()

    
    def parse_bnet(self, bnet_path):
        """
        Parse boolean network nodes and functions from .bnet format.
        """
        nodes, functions = [], []

        with open(bnet_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    if line.startswith("targets"):
                        continue
                    else:
                        node, func = line.strip().split(",")
                        nodes.append(node)
                        functions.append(func)

        return nodes, functions


    def get_neighbor_states(self, state):
        """
        Computes the states reachable from the given state in one step of synchronous/asynchronous update.
    
            Args:
                state (tuple[int, ...]): A tuple of 0s and 1s representing the Boolean network state.
    
            Returns:
                set[tuple[int, ...]]: A set of tuples of 0s and 1s representing the Boolean network states reachable
                    in one step from the given state.
        """
        reachable = set([])

        state_dict = {node: (self.__bool_algebra.TRUE if val == 1 else self.__bool_algebra.FALSE) for node, val in zip(self.list_of_nodes, state)}

        # Synchronous update
        if self.sync:
            new_state = []
            for i, fun in enumerate(self.functions):
                i_val = int(fun.subs(state_dict).simplify() == self.__bool_algebra.TRUE)
                new_state.append(i_val)
            reachable.add(tuple(new_state))

        # Asynchronous update
        else:
            for i, fun in enumerate(self.functions):
                i_val = int(fun.subs(state_dict).simplify() == self.__bool_algebra.TRUE)
                i_state = list(state)
                i_state[i] = i_val
                
                reachable.add(tuple(i_state))

        return reachable


    def generate_state_transition_system(self):
        """
        Generates the state transition system of the Boolean network.
    
            Returns:
                nx.DiGraph: NetworkX DiGraph object representing the state transition system.
    
        """
        G = nx.DiGraph()

        nodes = [self.__int_to_state(i) for i in range(2**self.num_nodes)]
        edges = set([])

        for node in nodes:
            neighbors = self.get_neighbor_states(node)
            for neighbor in neighbors:
                edges.add((self.__state_to_binary_str(node), self.__state_to_binary_str(neighbor)))

        G.add_nodes_from([self.__state_to_binary_str(node) for node in nodes])
        G.add_edges_from(edges)

        return G
    

    def get_attractors(self):
        """
        Computes the attractors of the Boolean network.
    
            Returns:
                list[set[tuple[int]]]: A list of sts attractors. Each attractor is a set of states.
        """
        sts = self.generate_state_transition_system()

        attractors = []
        for attractor in nx.attracting_components(sts):
            attractors.append(attractor)

        return attractors
    

    def generate_trajectory_dataset(
            self,
            filepath,
            n_trajectories = 1,
            trajectory_len = 10,
            sampling_frequency = 1,
            random_seed = 42,
        ):
        """
        Generate and save trajectory dataset to a chosen filepath.
        Args
            n_trajectories: number of trajectories in the dataset
            trajectory_len: lenght (num steps) for each trajectory
            sampling_frequency: sample a state for the dataset every (sampling_frequency) steps (sf=1 -> sample every step)
        """
        node_rows = [[str(node)] for node in self.list_of_nodes]
        exp_row = [""]
        trajectories = generate_random_walks(self.sts, n_trajectories, trajectory_len, seed=random_seed)
        # Attractor states count per trajectory
        att_counts = []
        for i, traj in enumerate(trajectories):
            att_counts.append(0)
            for j, state in enumerate(traj):
                if j%sampling_frequency == 0:
                    exp_row.append("EXP"+str(i)+":"+str(j//sampling_frequency))
                    for node_state, node_row in zip(state, node_rows):
                        node_row.append((node_state))
                    # Check if state is in an attractor
                    att_counts[i] += int(any([state in attractor for attractor in self.attractors]))
        
        # Save the dataset
        with open(filepath, "w") as f:
            f.write("\t".join(exp_row))
            for node_row in node_rows:
                f.write("\n" + "\t".join(node_row))

        # Check and save the percentage of attractor states per trajectory
        name, ext = os.path.splitext(filepath)
        att_filepath = name + "_attractors" + ext
        
        with open(att_filepath, "w") as f:
            # Header
            f.write("trajectory,attractor_state_percentage\n")
            for i in range(n_trajectories):
                f.write("EXP"+str(i)+",")
                f.write(str(att_counts[i]/(trajectory_len/sampling_frequency+1.0))+"\n")


    def save_graph_structure(self, path):
        """Save graph structure as list of edge pairs."""
        edges = sorted(self.graph_structure.edges())
        with open(path, "w") as f:
            for u, v in edges:
                f.write(u + " " + v + "\n")


def main(
    outpath,
    n_nodes = None,
    bnet_path = None,
    sync = False,
    allow_self_parent = False,
    n_trajectories = 1,
    trajectory_len = 10,
    sampling_frequency = 1,
    seed = 42,
    graph_structure_path = None,
):
    """Create a random boolean network of n_nodes and random boolean functions, then generate a trajectory dataset."""
    bn = BN(
        n_nodes=n_nodes,
        bnet_path=bnet_path,
        sync=sync,
        allow_self_parent=allow_self_parent,
        seed=seed,
    )
    
    bn.generate_trajectory_dataset(
        filepath=outpath,
        n_trajectories=n_trajectories,
        trajectory_len=trajectory_len,
        sampling_frequency=sampling_frequency,
        random_seed=seed,
    )

    if graph_structure_path:
        bn.save_graph_structure(graph_structure_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a random Boolean network and generate trajectory datasets.")

    parser.add_argument("-n", "--n-nodes", type=int, default=None, help="Number of nodes in the random Boolean network.")
    parser.add_argument("-b", "--bnet-path", type=str, default=None, help="A .bnet boolean network file (if n_nodes not provided)")
    parser.add_argument("-s", "--sync", action="store_true", help="Use synchronous updates (default: asynchronous).")
    parser.add_argument("-p", "--allow-self-parent", action="store_true", help="Allow nodes to have themselves as parents.")
    parser.add_argument("-r", "--seed", type=int, help="Random seed for network and trajectory generation.")
    parser.add_argument("-o", "--outpath", type=str, required=True, help="Output path for the generated trajectory dataset.")
    parser.add_argument("-t", "--n-trajectories", type=int, default=1, help="Number of trajectories to generate (default: 1).")
    parser.add_argument("-l", "--trajectory-len", type=int, default=10, help="Length of each trajectory (default: 10).")
    parser.add_argument("-f", "--sampling-frequency", type=int, default=1, help="Sampling frequency for trajectories (default: 1).")
    parser.add_argument("-g", "--graph-structure-path", type=str, help="Optional: path to save boolean network grapth structure.")

    args = parser.parse_args()

    main(**vars(args))
