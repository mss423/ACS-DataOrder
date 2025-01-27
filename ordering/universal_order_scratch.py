import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# -------------------------------------------------------------------
# Graph + Max Cover
# -------------------------------------------------------------------

def create_graph(data: np.ndarray, threshold: float) -> nx.Graph:
    """
    Build an undirected graph among the given 'data' points.
    Add an edge (i, j) iff cosine_similarity >= 'threshold'.
    """
    n = len(data)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    sims = cosine_similarity(data)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                G.add_edge(i, j)
    return G

def coverage_of_node(G: nx.Graph, node: int) -> set:
    """Coverage of 'node' = node plus its neighbors."""
    return {node} | set(G[node])

def run_max_cover(G: nx.Graph) -> list:
    """
    Greedy max cover on graph G, returning a list of clusters (sets of node indices).

    Steps:
      - While uncovered nodes exist:
        * pick the node whose coverage intersects uncovered the most
        * remove those covered from uncovered
      - Return the cluster sets
    """
    uncovered = set(G.nodes())
    clusters = []
    node_ids = []
    
    while uncovered:
        best_node = None
        best_cover = set()
        best_cover_size = -1
        
        for node in uncovered:
            c = coverage_of_node(G, node)
            c_int = c & uncovered
            if len(c_int) > best_cover_size:
                best_cover_size = len(c_int)
                best_cover = c_int
                best_node = node
        
        node_ids.append(best_node)
        clusters.append(best_cover)
        uncovered -= best_cover
    
    return clusters, node_ids

def pick_representative(G: nx.Graph, cluster: set) -> int:
    """
    Pick a single center (representative) from 'cluster':
    we choose the node with the largest coverage in G.
    Ties are broken arbitrarily (or by smallest node index).
    """
    best_node = None
    best_size = -1
    for nd in cluster:
        csize = len(coverage_of_node(G, nd))
        if csize > best_size or (csize == best_size and (best_node is None or nd < best_node)):
            best_size = csize
            best_node = nd
    return best_node

def hierarchical_max_cover(data: np.ndarray,
                           thresholds: list,
                           verbose: bool = True):
    """
    Builds a hierarchical clustering in the style requested:
    
    Steps:
      1) Start with threshold=1.0 on the full dataset => each point is its own cluster.
      2) Decrease threshold, run max cover => some number of clusters.
      3) Pick one 'center' for each cluster => only these centers remain 'active'.
      4) Decrease threshold further, build graph on these centers, run max cover again.
      5) Repeat until a single cluster covers all original data (or we exhaust thresholds).

    :param data:        (n_samples, n_features) array of your dataset
    :param thresholds:  A descending list of thresholds (starting at 1.0, then 0.95, etc.)
    :param verbose:     If True, prints out progress at each step.
    :return: A list of dictionaries, each describing one "round":
        [
          {
            'threshold': <current threshold>,
            'clusters': [set_of_nodes, set_of_nodes, ...],  # from run_max_cover
            'representatives': [node_index, node_index, ...],
            'active_map': [...],  # map from active node index -> original data index
          },
          ...
        ]
        Where each 'active_map' helps track which original data points were "active" at this threshold.
    """
    n = len(data)
    
    # Initially, the "active" set is all data points
    # We'll keep an array that maps "active node index" -> "original data index"
    active_map = np.arange(n)  # [0, 1, 2, ..., n-1]
    
    results = []
    
    for t in thresholds:
        # Build data array just for the active points
        current_data = data[active_map, :]
        
        # Create graph among these active points
        G_t = create_graph(current_data, t)
        
        # Run max cover => yields clusters (each cluster is a set of local indices in [0..len(active_map)-1])
        clusters_local = run_max_cover(G_t)
        
        # Convert local indices to original data indices, if desired
        clusters_original = []
        for clust in clusters_local:
            clusters_original.append({active_map[idx] for idx in clust})
        
        # Pick representatives for each cluster
        new_reps = []
        for clust in clusters_local:
            rep_idx = pick_representative(G_t, clust)
            new_reps.append(rep_idx)
        
        # Convert these local rep indices to original data indices
        new_reps_original = [active_map[r] for r in new_reps]
        
        # Collect results for this round
        round_info = {
            'threshold': t,
            'clusters': clusters_original,
            'representatives': new_reps_original,
            'active_map': active_map.copy()  # copy to record the old map
        }
        results.append(round_info)
        
        if verbose:
            print(f"\nThreshold={t:.2f}, #clusters={len(clusters_local)}")
            print("Clusters (original indices):")
            for i, corig in enumerate(clusters_original, start=1):
                print(f"  Cluster {i}: {sorted(corig)}")
            print("Chosen Representatives:", sorted(new_reps_original))
        
        # Check if we've ended up with just 1 cluster covering all points
        # i.e., if it covers 'len(active_map)' active nodes. 
        # But we also want to see if it covers the entire dataset if that is your condition:
        # That would require checking if clusters_local is 1 cluster of size == len(active_map),
        # AND that size is the entire dataset (?)
        
        # For now, let's do a simpler check: 
        # if there's exactly 1 cluster in clusters_local, we can see if that covers all active points:
        if len(clusters_local) == 1 and len(clusters_local[0]) == len(active_map):
            # We have a single cluster for the active set. 
            # If we want a single cluster for the entire original dataset, we can also check:
            if len(active_map) == n:
                if verbose:
                    print("All data are covered by a single cluster. Stopping early.")
                break
        
        # Update the active set to be the new reps
        active_map = np.array(new_reps_original)
        
        # If there's only 1 rep left, no further merges are possible, so we can stop as well
        if len(active_map) == 1:
            if verbose:
                print("Reached a single representative, stopping.")
            break
    
    return results

def build_total_order(hierarchy):
    """
    Convert the multi-round 'hierarchy' (as returned by hierarchical_max_cover)
    into a single linear ordering of all data points.

    We'll do a "top-down" pass:
      - Start from the final round's clusters (the 'highest' level).
      - For each cluster, find which reps in the previous round contributed to it,
        then recursively expand those subclusters, and so on,
        until we reach the earliest layer (likely singletons).
    Because we discard non-representative points at each round, the clusters are not
    strictly nested. This code attempts to track back as best as possible.
    """
    if not hierarchy:
        return []

    # A helper function to find which cluster(s) in 'round_info'
    # contain a given representative r.
    # Returns a list, since in rare cases a rep might appear in >1 cluster
    def find_cluster_of_rep(round_info, rep):
        # round_info['clusters'] is a list of sets of original indices
        # We want to see which set(s) contain 'rep'.
        found = []
        for cset in round_info['clusters']:
            if rep in cset:
                found.append(cset)
        return found

    # We'll store all data points in a list, but avoid duplicates with a visited set
    visited = set()
    ordering = []

    def expand_cluster(cluster_set, round_idx):
        """
        Recursively expand 'cluster_set' from the cluster in round_idx
        to its children in round_idx - 1, if any.
        """
        # First, add all points from this cluster that are not yet visited
        new_points = [p for p in sorted(cluster_set) if p not in visited]
        ordering.extend(new_points)
        for p in new_points:
            visited.add(p)

        if round_idx == 0:
            # We are at the earliest round, no deeper expansions
            return
        
        # Otherwise, see which reps from (round_idx - 1) contributed to this cluster.
        # round_idx - 1 means the previous round in 'hierarchy'.
        # The representatives for that round are in:  hierarchy[round_idx - 1]['representatives']
        # BUT we also need to see which cluster(s) those reps came from in round_idx - 1.
        prev_info = hierarchy[round_idx - 1]
        prev_reps = prev_info['representatives']  # e.g. [orig_idx_1, orig_idx_2, ...]
        
        # We look for all reps in 'prev_reps' that appear in cluster_set
        # Because cluster_set is made of "original data indices," if rep in cluster_set,
        # that means it contributed to forming the cluster at round_idx.
        reps_in_this_cluster = [r for r in prev_reps if r in cluster_set]

        # For each such rep, find the actual subcluster from round_idx-1 that rep came from
        for rep in reps_in_this_cluster:
            child_clusters = find_cluster_of_rep(prev_info, rep)
            # Typically 'child_clusters' should be exactly one set. If multiple, we can expand them all.
            for child_set in child_clusters:
                expand_cluster(child_set, round_idx - 1)

    # Start from the final round
    final_idx = len(hierarchy) - 1
    final_info = hierarchy[final_idx]

    # In the final round, we typically have a few clusters covering all data
    # Sort them by descending size
    final_clusters = sorted(final_info['clusters'], key=lambda s: len(s), reverse=True)

    # Expand each final cluster in order
    for cset in final_clusters:
        expand_cluster(cset, final_idx)

    return ordering


def kmeans_order(data, n_clusters=None, random_state=42):
    """
    Baseline ordering algorithm:
      1. Run k-means on the data with 'n_clusters'.
      2. For each cluster, pick one 'representative' data point 
         (the one closest to the centroid).
      3. Put these representatives first in the ordering.
      4. Then, for the remaining points, sort them by their distance 
         to their cluster centroid in ascending order.

    :param data:         2D array, shape (n_samples, n_features).
    :param n_clusters:   Number of clusters for k-means.
    :param random_state: For reproducibility of k-means.
    :return:             A list of data-point indices in the desired order.
    """
    n = len(data)
    if not n_clusters:
        n_closers = data.shape[1]
    
    # (A) Fit k-means to the entire dataset
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_  # shape: (k, n_features)
    labels = kmeans.labels_              # shape: (n, )

    # (B) Find the representative data point for each cluster
    #     i.e. the one whose distance to the centroid is minimal.
    # We can compute a (n x k) distance matrix to all centroids, or simply 
    # filter points by their label.
    
    # Distances from each data point to each centroid
    # shape = (n, k)
    dist_matrix = cdist(data, centroids, metric='euclidean')
    
    cluster_reps = [-1] * n_clusters  # will store the index of the rep for each cluster
    for cluster_id in range(n_clusters):
        # Find points belonging to this cluster
        cluster_points_idx = np.where(labels == cluster_id)[0]
        
        # Among these, pick the index with the smallest distance to cluster_id centroid
        dists = dist_matrix[cluster_points_idx, cluster_id]
        best_local_idx = np.argmin(dists)
        
        # actual index in the dataset
        rep_idx = cluster_points_idx[best_local_idx]
        
        cluster_reps[cluster_id] = rep_idx

    # (C) Build the final ordering:
    #     Step 1: Add all cluster representatives (unique indices).
    #             We'll do it in ascending cluster_id for consistency.
    ordering = list(cluster_reps)
    
    # (D) For the remaining points, sort them by distance to their *own* centroid.
    #     We'll exclude the reps so we don't double-list them.
    reps_set = set(cluster_reps)
    
    # Prepare a list of (point_idx, distance_to_its_cluster_centroid)
    dist_to_centroid = []
    for idx in range(n):
        if idx not in reps_set:
            c_id = labels[idx]
            dist_val = dist_matrix[idx, c_id]
            dist_to_centroid.append((idx, dist_val))
    
    # Sort by ascending distance
    dist_to_centroid.sort(key=lambda x: x[1])
    
    # (E) Append these points in ascending distance
    ordering.extend(pt_idx for (pt_idx, dist_val) in dist_to_centroid)
    
    return ordering


