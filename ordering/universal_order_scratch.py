import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_graph(data: np.ndarray, percentile: float = 90.0) -> nx.Graph:
    """
    Build an undirected graph from 'data' based on pairwise cosine similarities.
    
    :param data: 2D array of shape (n_samples, n_features)
    :param percentile: The percentile of similarity above which an edge is created.
                       For example, 90.0 means we keep edges for the top 10% of similarities.
    :return: A NetworkX Graph where each node represents an index in 'data'
             and an edge exists if the similarity exceeds the chosen threshold.
    """
    # 1) Compute the full cosine similarity matrix (n x n)
    sims = cosine_similarity(data)  # sims[i, j] in [-1, 1] for cosine
    
    # 2) Determine the similarity threshold by looking at off-diagonal entries
    #    Flatten only i<j or i>j to avoid the diagonal (which is 1.0 for each i,i).
    n = len(data)
    upper_tri_indices = np.triu_indices(n, k=1)  # (row indices, col indices) for i<j
    sim_values = sims[upper_tri_indices]
    
    # 3) Get the cutoff similarity at the given percentile
    cutoff = np.percentile(sim_values, percentile)
    
    # 4) Build the graph
    G = nx.Graph()
    G.add_nodes_from(range(n))  # Add all nodes first
    
    # 5) Add edges for pairs above the similarity cutoff
    for (i, j), sim_val in zip(zip(upper_tri_indices[0], upper_tri_indices[1]), sim_values):
        if sim_val >= cutoff:
            G.add_edge(i, j)
            
    return G

def coverage_of_cluster(G: nx.Graph, cluster_nodes: set) -> set:
    """
    Given a set of node indices (cluster_nodes) in Graph G,
    return the 'covered' set: the cluster itself plus all its neighbors.
    """
    covered = set(cluster_nodes)
    for u in cluster_nodes:
        covered.update(G[u])  # neighbors of u
    return covered

def hierarchical_max_coverage_order(G: nx.Graph) -> list:
    """
    1) Builds a hierarchy (binary merge tree) of clusters via greedy max-coverage merges.
    2) Flattens that hierarchy in a coverage-aware order, returning a single list of node indices.
    """
    # --- Step A: Initialize each node as its own cluster ---
    clusters = [{node} for node in G.nodes()]
    
    # We'll store the merge tree in a dict:
    #   merge_tree[frozenset_of_nodes] = (childA, childB)
    # For leaves, store (None, None).
    merge_tree = {}
    for c in clusters:
        merge_tree[frozenset(c)] = (None, None)  # Leaf cluster
    
    # --- Step B: Repeatedly merge pairs of clusters that yield the largest coverage union ---
    while len(clusters) > 1:
        best_pair = None
        best_cover_size = -1
        
        # Naive O(n^2) search over pairs
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                union_cluster = c1.union(c2)
                cover = coverage_of_cluster(G, union_cluster)
                if len(cover) > best_cover_size:
                    best_cover_size = len(cover)
                    best_pair = (i, j, union_cluster)
        
        # best_pair has (index1, index2, union_of_their_nodes)
        i, j, union_set = best_pair
        new_cluster = frozenset(union_set)
        
        # Record children in the merge tree
        old_c1 = frozenset(clusters[i])
        old_c2 = frozenset(clusters[j])
        merge_tree[new_cluster] = (old_c1, old_c2)
        
        # Remove old clusters and add the merged one
        for idx in sorted([i, j], reverse=True):
            clusters.pop(idx)
        clusters.append(set(union_set))
    
    # Now exactly one final cluster remains
    root = frozenset(clusters[0])
    
    # --- Step C: Flatten the merge tree into a list of node indices ---
    ordering = []
    
    def flatten(node: frozenset, covered_so_far: set):
        """
        Recursively collect nodes from 'node' in an order that picks
        the child with the larger coverage gain first.
        """
        children = merge_tree[node]
        # If leaf:
        if children == (None, None):
            # node is a frozenset of (likely) 1 element
            ordering.extend(list(node))
            return
        
        left, right = children
        left_cover  = coverage_of_cluster(G, left)  - covered_so_far
        right_cover = coverage_of_cluster(G, right) - covered_so_far
        
        # Decide which child first
        if len(left_cover) >= len(right_cover):
            flatten(left, covered_so_far)
            covered_left = covered_so_far.union(coverage_of_cluster(G, left))
            flatten(right, covered_left)
        else:
            flatten(right, covered_so_far)
            covered_right = covered_so_far.union(coverage_of_cluster(G, right))
            flatten(left, covered_right)
    
    flatten(root, set())
    print(len(ordering))
    return ordering

def create_graph(data: np.ndarray, threshold: float) -> nx.Graph:
    """
    Build an undirected graph from 'data' based on pairwise cosine similarities.
    Add an edge (i, j) iff the similarity >= threshold.
    
    :param data:       2D array (n_samples, n_features)
    :param threshold:  Similarity threshold for edge creation
    :return:           NetworkX Graph with edges for pairs above threshold
    """
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))  # Add all nodes
    
    # Compute full pairwise cosine similarity
    sims = cosine_similarity(data)
    
    # Add edges where similarity >= threshold
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                G.add_edge(i, j)
    
    return G

def coverage_of_node(G: nx.Graph, node: int) -> set:
    """
    'Coverage' of a node is the node itself plus its neighbors in G.
    """
    return {node} | set(G[node])  # node plus its neighbors

def run_max_cover(G: nx.Graph):
    """
    A simple (greedy) max cover:
      - We repeatedly pick the node whose 'coverage' (itself + neighbors)
        covers the most *uncovered* nodes.
      - Then we remove all those covered nodes from the pool.
      - Continue until no uncovered nodes remain.
    Returns a list of clusters (each cluster is a Python set of nodes).
    """
    uncovered = set(G.nodes())
    clusters = []
    
    while uncovered:
        best_node = None
        best_cover_size = -1
        best_cover = set()
        
        # Among uncovered nodes, pick the one whose coverage set
        # intersects with uncovered the most
        for node in uncovered:
            c = coverage_of_node(G, node)
            c_intersect = c & uncovered
            if len(c_intersect) > best_cover_size:
                best_cover_size = len(c_intersect)
                best_node = node
                best_cover = c_intersect
        
        # best_node yields best coverage among the uncovered
        clusters.append(best_cover)
        # Remove these covered nodes from the pool
        uncovered -= best_cover
    
    return clusters

def multi_threshold_max_cover(data: np.ndarray, thresholds: list):
    """
    For each threshold in descending order:
      1) Create a graph with that threshold,
      2) Run 'max cover' clustering,
      3) Store/print the resulting clusters.
    """
    threshold_to_clusters = {}
    for t in thresholds:
        print(f"\n=== Threshold = {t:.2f} ===")
        
        # Build the graph at this threshold
        G_t = create_graph(data, t)
        
        # Run the simple max cover
        clusters = run_max_cover(G_t)
        
        # Display the result
        print(f"Graph has {G_t.number_of_nodes()} nodes and {G_t.number_of_edges()} edges.")
        print(f"Number of clusters found: {len(clusters)}")
        for i, c in enumerate(clusters, 1):
            print(f"  Cluster {i}: {sorted(c)}")
        
        threshold_to_clusters[t] = clusters
    
    return threshold_to_clusters

# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small synthetic dataset: 10 samples, 5 features
    np.random.seed(42)
    data = np.random.rand(10, 5)
    
    # Build a similarity graph, keep edges in top 85% similarity
    G = create_similarity_graph(data, percentile=85.0)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Compute the hierarchical max-coverage order
    universal_order = hierarchical_max_coverage_order(G)
    print("Universal ordering of nodes:", universal_order)
