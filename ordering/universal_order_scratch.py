import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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
    def __init__(self, center_id, indices, size=1):
        self.indices = indices  # which rows of the *original* data are in this node
        self.center_id = center_id
        self.size = size

def create_graph(data: np.ndarray, threshold: float, labels: np.ndarray = None) -> nx.Graph:
    """
    Build an undirected graph among the given 'data' points.
    Add an edge (i, j) iff cosine_similarity >= 'threshold'.
    """
    n = len(data)
    G = nx.Graph()
    if not labels:
        G.add_nodes_from(range(n))
        labels = range(n)
    else:
        G.add_nodes_from(labels)
    
    sims = cosine_similarity(data)
    for i in labels:
        for j in labels:
            if i == j:
                continue
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

def hierarchical_max_cover(data, threshold=1.0, step=0.05, lb=0.7):
    cos_sim = cosine_similarity(data)
    G = create_graph(data, threshold=threshold)

    level = 0
    hierarchy = {}
    base_layer = []
    for i in range(len(data)):
        base_layer.append(ClusterNode(center_id=i, indices=[i]))
    hierarchy[level] = base_layer

    while len(hierarchy[level]) > 1 and threshold >= lb:
        level += 1
        hierarchy[level] = {}
        clusters, node_ids = run_max_cover(G)
        new_level = []
        for cluster, node_id in zip(clusters, node_ids):
            children = list(cluster)
            new_level.append(ClusterNode(center_id=node_id, indices=children))
        hierarchy[level] = new_level
        
        threshold -= step
        G = create_graph(data[:, node_ids], threshold, labels=node_ids)

    return hierarchy

def total_order(hierarchy):
    l = len(hierarchy)
    order = []
    for level in range(l-1, -1, -1):
        for node in hierarchy[level]:
            if node.center_id not in order:
                order.append(node.center_id)
    return order