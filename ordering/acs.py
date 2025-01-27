import networkx as nx
import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def build_graph(cos_sim, sim_thresh=0.0, max_degree=None, labels=None):
    G = nx.Graph()
    for i in range(len(cos_sim)):
        G.add_node(i)
        # Sort neighbors by similarity in descending order
        neighbors = sorted(enumerate(cos_sim[i]), key=lambda x: x[1], reverse=True)
        for j, similarity in neighbors:
            if j == i:
                continue
            if max_degree and G.degree(i) >= max_degree:
                break  # Exit the inner loop if max_degree is reached
            if similarity >= sim_thresh:
                G.add_edge(i, j, weight=similarity)
        # add self-loop, doesn't count toward max_degree
        G.add_edge(i, i, weight=1)
    return G

def calculate_similarity_threshold(data, num_samples, coverage, cap=None, epsilon=None, labels=None, sims=[707,1000]):
    total_num = len(data)
    if epsilon is None:
        # There is a chance that we never get close enough to "coverage" to terminate
        # the loop. I think at the very least we should have epsilon > 1/total_num.
        # So let's set set epsilon equal to the twice of the minimum possible change
        # in coverage.
        epsilon = 5 * 10 / total_num  # Dynamic epsilon

    if coverage < num_samples / total_num:
        node_graph = build_graph(data, 1)
        samples, rem_nodes = max_cover(node_graph, num_samples)
        return 1, node_graph, samples
    # using an integer for sim threhsold avoids lots of floating drama!
    sim_upper = sims[1]
    sim_lower = sims[0] # 707 corresponds to 0.707
    max_run = 20
    count = 0
    current_coverage = 0

    # Set sim to sim_lower to run the first iteration with sim_lower. If we
    # cannot achieve the coverage with sim_lower, then return the samples.
    sim = (sim_upper + sim_lower) / 2
    # node_graph = build_graph(data, sim / 1000, labels=labels
    cap = (2 * total_num * coverage) / num_samples
    while abs(current_coverage - coverage) > epsilon and sim_upper - sim_lower > 1:
        if count >= max_run:
            print(f"Reached max number of iterations ({max_run}). Breaking...")
            break
        count += 1

        node_graph = build_graph(data, sim / 1000, max_degree=cap, labels=labels)
        samples, rem_nodes = max_cover(node_graph, num_samples)
        current_coverage = (total_num - rem_nodes) / total_num

        if current_coverage < coverage:
            sim_upper = sim
        else:
            sim_lower = sim
        sim = (sim_upper + sim_lower) / 2
    # print(f"Converged to tau = {sim/1000}")
    return sim / 1000, node_graph, samples

def max_cover(graph, k):
    nodes = list(graph.nodes())
    n = len(nodes)
    selected_nodes = []
    covered_nodes = set()

    # Get list of singletons and remove from contention
    # node_to_id = {node: i for i, node in enumerate(graph.nodes())}  
    # singleton_node_ids = [node_to_id[node] for node in graph.nodes() if graph.degree(node) == 0]
    # print(f"Number of singletons = {len(singleton_node_ids)}")
    # covered_nodes.update(singleton_node_ids)

    for _ in range(k):
        if not nodes:
            break
        max_cover_node = max([node for node in nodes if node not in covered_nodes],
            key=lambda n: len([neighbor for neighbor in graph.neighbors(n) if neighbor not in covered_nodes])
            )
        selected_nodes.append(max_cover_node)
        covered_nodes.add(max_cover_node)
        covered_nodes.update(graph.neighbors(max_cover_node))

        # Remove neighbors of selected node
        for neighbor in graph.neighbors(max_cover_node):
            if neighbor in nodes:
                nodes.remove(neighbor)
    return selected_nodes, len(nodes)





class ClusterNode:
    """
    A node in our hierarchical ACS tree.
    Attributes:
        indices: np.ndarray of shape (m,)
            The indices (into the *original* dataset) for all points in this node's subtree.
        center_indices: np.ndarray of shape (k,)
            Indices (into `indices`) of the chosen cluster centers at this level.
            So the actual center in the original dataset is indices[center_indices[i]].
        children: list of ClusterNode
            The subnodes (one child per chosen center).
    """
    def __init__(self, indices):
        self.indices = indices  # which rows of the *original* data are in this node
        self.center_indices = None
        self.children = []

def assign_points_to_centers(node_graph, subset_indices, samples):
    """
    Given a built node_graph for the current subset, and the 'samples' (the chosen centers),
    figure out which points are covered by which center.

    Parameters
    ----------
    node_graph : adjacency info or similar structure
        Must be consistent with the current subset of data.
    subset_indices : np.ndarray
        The actual (original) indices of the data in this subset.
    samples : np.ndarray
        Indices (0-based w.r.t. subset) of the chosen centers.

    Returns
    -------
    membership : list of np.ndarray
        membership[i] is an array of *subset indices* belonging to center i.
        That includes the center itself plus any points it covers.

    NOTE: If a point is adjacent to multiple centers, we can choose the first or
    do a more refined assignment. This is a simple approach.
    """
    k = len(samples)
    membership = [[] for _ in range(k)]

    # For convenience, turn 'samples' into a set for quick membership checks
    sample_set = set(samples)

    # We assume `node_graph[u]` gives neighbors of u. 
    # If your node_graph is stored differently, adapt as needed.
    # We'll do a naive approach: for each point u in this subset, see which center(s) it's adjacent to.
    num_points = len(subset_indices)
    for u in range(num_points):
        if u in sample_set:
            # This point is itself a center
            center_index = samples.index(u) if isinstance(samples, list) else np.where(samples == u)[0][0]
            membership[center_index].append(u)
        else:
            # Not a center. Check adjacency to see if it's covered by a center.
            neighbors = node_graph[u]  # list of adjacent nodes (in 0..num_points-1)
            # Find any overlap with sample_set
            possible_centers = sample_set.intersection(neighbors)
            if possible_centers:
                # Just pick the first center in possible_centers (arbitrary)
                chosen_center = next(iter(possible_centers))
                center_index = samples.index(chosen_center) if isinstance(samples, list) else np.where(samples == chosen_center)[0][0]
                membership[center_index].append(u)
            else:
                # If it's not adjacent to any center, it remains uncovered,
                # or we can assign it to a "nearest center" by some rule.
                # For now, let's skip it or put it in a "closest" center artificially.
                pass

    # Convert to arrays
    membership = [np.array(m) for m in membership]
    return membership

def build_hierarchy_tree(data, node: ClusterNode, coverage=0.9, cap=None, epsilon=None, labels=None, sims=[0, 1000]):
    """
    Recursive function that:
      - Applies ACS to the 'node' subset (node.indices).
      - Picks half as many centers as there are points in 'node'.
      - Builds children (subclusters) and recurses.
    """
    m = len(node.indices)
    if m <= 1:
        # A single point or none; no further subdivision
        return

    k = m // 2  # how many centers we want from this subset
    if k < 1:
        return

    # Gather the actual data for this subset
    subset_data = data[node.indices]  # shape: (m, d)

    # Run ACS on subset_data
    print(k)
    sim_thresh, node_graph, samples = calculate_similarity_threshold(
        data=subset_data,
        num_samples=k,
        coverage=coverage,
        cap=cap,
        epsilon=epsilon,
        labels=labels,
        sims=sims
    )
    # 'samples' are indices in [0..m-1] referring to 'subset_data'
    # We'll store them, but we must keep in mind these are relative to node.indices
    node.center_indices = samples  # store subset-based indices

    # Next, figure out membership: which sub-points get assigned to each center
    # This requires adjacency. We assume node_graph[u] = list of neighbors of u
    membership = assign_points_to_centers(node_graph, node.indices, list(samples))

    # Create child nodes. Each chosen center spawns one child with all points covered by it.
    for center_idx, member_subset_indices in enumerate(membership):
        if len(member_subset_indices) == 0:
            continue

        child_node = ClusterNode(indices=node.indices[member_subset_indices])
        node.children.append(child_node)

        # Recurse
        build_hierarchy_tree(data, child_node, coverage, cap, epsilon, labels, sims)

def hierarchical_acs_tree(data, coverage=0.8, cap=None, epsilon=None, labels=None, sims=[707, 1000]):
    """
    Top-level function: build a hierarchical ACS tree from the entire dataset.
    """
    root = ClusterNode(indices=np.arange(len(data)))  # root covers all data
    build_hierarchy_tree(data, root, coverage, cap, epsilon, labels, sims)
    return root


def dfs_order(node: ClusterNode):
    """
    Returns a list of the original data indices in a DFS order
    according to the hierarchy. Points in the same cluster appear contiguously.
    """
    # If no children, just return the node's indices (base case)
    if len(node.children) == 0:
        return list(node.indices)

    # Otherwise, DFS into each child in order
    ordering = []
    for child in node.children:
        ordering.extend(dfs_order(child))

    return ordering

def get_total_ordering(root: ClusterNode):
    """
    A convenient wrapper that starts DFS from the root
    and returns a single ordering of all data points.
    """
    return dfs_order(root)

# def hierarchical_acs(data, coverage=0.9, cap=None, epsilon=None, labels=None, sims=[0, 1000]):
#     """
#     Perform a hierarchical ACS in a top-down manner:
#       1. From the original data of size n, pick n//2 cluster centers.
#       2. From those n//2 centers, pick n//4 centers.
#       3. Continue until you reach 1 center or can't halve anymore.

#     Parameters
#     ----------
#     data : np.ndarray
#         The full dataset you want to cluster. Shape (n, d).
#     coverage : float, optional
#         Desired coverage fraction for each ACS level.
#     cap : float or None, optional
#         Passed to `calculate_similarity_threshold` for max_degree. If None, no explicit cap is used.
#     epsilon : float or None, optional
#         Stopping threshold for the coverage.
#     labels : array-like or None, optional
#         Optional labels if `build_graph` uses them for building edges.
#     sims : list of two ints, optional
#         Lower and upper bounds for the similarity threshold in integer form (e.g. [707, 1000]).

#     Returns
#     -------
#     hierarchy : list of dict
#         A list where each element is a dictionary describing one level of the hierarchy:
#         {
#           "level": int,
#           "num_samples": int,
#           "selected_indices": np.ndarray of shape (k, ),
#           "selected_data": np.ndarray of shape (k, d),
#           "similarity_threshold": float
#         }
#         The final entry typically has a single center if n is a power of 2, or a small set otherwise.
#     """
#     n = len(data)
#     level_data = data  # Start with all points
#     level = 0

#     hierarchy = []

#     while len(level_data) > 1:
#         # Determine how many samples to pick in this level (integer division).
#         # If n is not a perfect power of 2, eventually this might become 0 or 1.
#         k = len(level_data) // 2
#         if k < 1:
#             # Can't pick fewer than 1 center, so we stop.
#             break

#         # Run ACS on the current subset (level_data) to get k centers.
#         sim_thresh, node_graph, samples = calculate_similarity_threshold(
#             data=level_data,
#             num_samples=k,
#             coverage=coverage,
#             cap=cap,
#             epsilon=epsilon,
#             labels=labels,
#             sims=sims
#         )

#         # 'samples' should be the indices (relative to level_data) of the chosen cluster centers.
#         selected_data = level_data[samples]

#         # Store hierarchy info
#         hierarchy.append({
#             "level": level,
#             "num_samples": k,
#             "selected_indices": samples,
#             "selected_data": selected_data,
#             "similarity_threshold": sim_thresh
#         })

#         # Now, move to the next level: the newly selected centers become our "data"
#         level_data = selected_data
#         level += 1

#     return hierarchy
