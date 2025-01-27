import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

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
        
        clusters.append(best_cover)
        uncovered -= best_cover
    
    return clusters

def pick_representative(G: nx.Graph, cluster: set) -> int:
    """
    Among the 'cluster' nodes, pick a single 'representative':
    the node with the largest coverage (tie-break by smallest ID).
    """
    best_node = None
    best_size = -1
    for nd in cluster:
        cov_size = len(coverage_of_node(G, nd))
        if cov_size > best_size or (cov_size == best_size and (best_node is None or nd < best_node)):
            best_size = cov_size
            best_node = nd
    return best_node


# -------------------------------------------------------------------
# Data Structures for the Multi-Layer Hierarchy
# -------------------------------------------------------------------

class ClusterNode:
    """
    Each cluster node stores:
      - 'members': a set of original data indices (integers).
      - 'parent': a pointer to the cluster node in the *next* layer 
                  that it merges into (or None if it's in the final layer).
    """
    def __init__(self, members):
        self.members = members  # set of original data indices
        self.parent = None      # link to a ClusterNode in the next layer

def multi_round_max_cover_with_reps_backlinks(data: np.ndarray, thresholds: list):
    """
    For each threshold t in descending order:
      1) We treat the "current_layer" as the active clusters (one node each).
      2) Build a graph among them by computing a 'representative vector' for each cluster
         (e.g., the centroid of its members).
      3) Run max cover on that graph => sets of cluster-nodes that merge.
      4) For each merged set, create a new cluster node that is the union of all 
         child nodes' .members, and set child_node.parent = that new cluster node.

    Returns a list 'layers', where layers[i] is the list of ClusterNode objects 
    at threshold=thresholds[i], and layers[-1] is the final layer (lowest threshold).
    """
    n = len(data)
    # --- Stage 0: each data point is its own cluster node
    current_layer = []
    for i in range(n):
        cnode = ClusterNode(members={i})
        current_layer.append(cnode)
    
    layers = []
    
    for idx, t in enumerate(thresholds):
        # Build array for current layer's "representatives" to measure similarity
        node_map = list(current_layer)   # local_index -> cluster node
        rep_data = []
        
        # For each cluster node, compute a centroid of its .members in the original data
        for cnode in node_map:
            arr = data[list(cnode.members), :]  # sub-array
            centroid = arr.mean(axis=0)
            rep_data.append(centroid)
        rep_data = np.vstack(rep_data)
        
        # Build graph among these cluster-node centroids
        G_t = create_graph(rep_data, t)
        
        # Run max cover => merges sets of local indices
        clusters_t = run_max_cover(G_t)
        
        # Build new layer of cluster nodes
        new_layer = []
        
        for cset in clusters_t:
            # Union all the original data from children
            union_members = set()
            for local_idx in cset:
                union_members |= node_map[local_idx].members
            
            # Create the "parent" cluster node
            parent_node = ClusterNode(members=union_members)
            new_layer.append(parent_node)
            
            # Link each child node's parent to this new node
            for local_idx in cset:
                node_map[local_idx].parent = parent_node
        
        # Store the current layer (these are the clusters at threshold=t)
        layers.append(current_layer)
        
        # Move on
        current_layer = new_layer
    return layers


# -------------------------------------------------------------------
# Building a Hierarchical Ordering from the Final Layer
# -------------------------------------------------------------------

def get_children_of_node(cnode, prev_layer):
    """
    Among the clusters in 'prev_layer', find those whose .parent == cnode.
    """
    return [child for child in prev_layer if child.parent == cnode]

def expand_cluster_hier(cnode, layer_index, layers):
    """
    Recursively expand 'cnode' by looking at the previous layer (layer_index-1),
    finding all child clusters that point to cnode, 
    sorting them (e.g. by size) and expanding them in a DFS or BFS manner.
    
    If layer_index == 0, cnode is in the first layer, so it has no children => 
    just return the single data points in cnode.members (sorted).
    
    Otherwise:
      - gather all children
      - sort them by some criterion (like descending size)
      - expand each child
    """
    if layer_index == 0:
        # No children => just return these members sorted
        return sorted(cnode.members)
    
    prev_layer = layers[layer_index - 1]
    # Find child nodes that merged into cnode
    children = get_children_of_node(cnode, prev_layer)
    
    # Sort children by descending size
    children_sorted = sorted(children, key=lambda ch: len(ch.members), reverse=True)
    
    ordering = []
    for child_node in children_sorted:
        # Recursively expand
        ordering.extend(expand_cluster_hier(child_node, layer_index - 1, layers))
    
    return ordering

def alternative_1_hierarchical_order(layers):
    """
    A refined Alternative 1:
      - Look at the *final layer* (layers[-1]) => top-level clusters at the lowest threshold
      - Sort them by size (descending)
      - For each final cluster, do a hierarchical expansion that visits its children 
        in descending size, and so on, down to the first layer (threshold=1.0).
      
    Returns a single list of original data point indices in a "top-down" order.
    """
    if not layers:
        return []
    
    final_layer = layers[-1]
    final_idx = len(layers) - 1
    
    # Sort final clusters by descending size
    final_layer_sorted = sorted(final_layer, key=lambda c: len(c.members), reverse=True)
    
    overall_order = []
    for top_cnode in final_layer_sorted:
        expanded = expand_cluster_hier(top_cnode, final_idx, layers)
        overall_order.extend(expanded)
    
    return overall_order


# -------------------------------------------------------------------
# Alternative 2: (unchanged) disregard the hierarchy,
#  just build a graph of all original points at final threshold
#  and sort them by coverage.
# -------------------------------------------------------------------
def alternative_2_ordering_all_data(data: np.ndarray, final_thresh: float) -> list:
    """
    Alternative 2:
      - Ignore the multi-layer merges.
      - Build a graph on ALL original data at final_thresh,
      - Rank each data point by coverage size (descending).
    """
    G_final = create_graph(data, final_thresh)
    
    coverage_list = []
    for nd in G_final.nodes():
        csize = len(coverage_of_node(G_final, nd))
        coverage_list.append((nd, csize))
    
    # Sort descending by coverage
    coverage_list.sort(key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in coverage_list]


# -------------------------------------------------------------------
# DEMO
# -------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    # Example dataset
    data = np.random.rand(10, 4)  # 10 points, 4 features
    
    thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    
    # 1) Build multi-layer structure
    layers = multi_round_max_cover_with_reps_backlinks(data, thresholds)
    
    # Print summary
    print("\n--- Layers Summary ---")
    for i, layer in enumerate(layers):
        if i < len(thresholds):
            th = thresholds[i]
            print(f"Layer {i} (threshold={th:.2f}): #clusters={len(layer)}")
        else:
            print(f"Final layer (post threshold={thresholds[-1]}): #clusters={len(layer)}")
        for idx, cnode in enumerate(layer, start=1):
            print(f"  ClusterNode {idx}, size={len(cnode.members)}, members={sorted(cnode.members)}")
    
    # 2) Alternative 1: hierarchical expansion from final clusters
    alt1_order = alternative_1_hierarchical_order(layers)
    
    # 3) Alternative 2: ignoring the hierarchy
    final_t = thresholds[-1]
    alt2_order = alternative_2_ordering_all_data(data, final_t)
    
    print("\n==============================================")
    print("Alternative 1: Hierarchical Ordering from Final Layer (Largest final cluster first)")
    print("Resulting order of all data points:", alt1_order)
    
    print("\nAlternative 2: All data by coverage in final graph")
    print("Resulting order of all data points:", alt2_order)
