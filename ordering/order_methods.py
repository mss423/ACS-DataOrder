from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vertex_embed import get_embeddings_task

def incremental_kmeans_ordering(data, Ks):
    """
    Orders data using an incremental k-means approach.

    Args:
        data: The input data to be ordered.
        k_values: A list of k values to use for clustering.

    Returns:
        A list of data indices representing the ordered data.
    """
    ordered_indices = []
    centroids = []  # Store previously found centroids

    for k in Ks:
        # Initialize KMeans with previous centroids (if any)
        if len(centroids) > 0:
            # 1. Select data points closest to previous centroids
            closest_indices = [np.argmin(np.linalg.norm(data - centroid, axis=1)) for centroid in centroids]

            # 2. Add new random centroids
            num_new_centroids = k - len(centroids)
            if num_new_centroids > 0:
                remaining_indices = np.setdiff1d(np.arange(data.shape[0]), closest_indices)
                new_centroid_indices = np.random.choice(remaining_indices, size=num_new_centroids, replace=False)
                closest_indices.extend(new_centroid_indices)

            # 3. Combine previous and new centroids
            initial_centroids = data[closest_indices]
            kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1)
        else:
            kmeans = KMeans(n_clusters=k)

        kmeans.fit(data)
        centroids = kmeans.cluster_centers_  # Update centroids

        # Find the data point closest to each centroid
        closest_indices = []
        for centroid in centroids:
            distances = np.linalg.norm(data - centroid, axis=1)  # Calculate distances
            closest_index = np.argmin(distances)  # Find index of closest point

            # if there is a tie, pick first one
            if closest_index in ordered_indices:
                closest_index = np.where(distances == distances[closest_index])[0][0]
            closest_indices.append(closest_index)

        # Add new indices to the ordered list (while maintaining order)
        for index in closest_indices:
            if index not in ordered_indices:
                ordered_indices.append(index)

    return ordered_indices

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
            if similarity >= sim_thresh:# and labels and labels[i]==labels[j]:
                G.add_edge(i, j, weight=similarity)
        # add self-loop, doesn't count toward max_degree
        G.add_edge(i, i, weight=1)
    return G

# Graph sampling algorithms (max-cover)
def max_cover_sampling(graph, k):
    nodes = list(graph.nodes())
    selected_nodes = set()
    covered_nodes = set()

    for _ in range(k):
      if not nodes:
        break
      max_cover_node = max(
            [node for node in nodes if node not in covered_nodes],
            key=lambda n: len([neighbor for neighbor in graph.neighbors(n) if neighbor not in covered_nodes])
        )
      selected_nodes.add(max_cover_node)
      covered_nodes.add(max_cover_node)
      covered_nodes.update(graph.neighbors(max_cover_node))

      # Remove neighbors of selected node
      for neighbor in graph.neighbors(max_cover_node):
        if neighbor in nodes:
          nodes.remove(neighbor)
    return list(selected_nodes), len(nodes)

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
        samples, rem_nodes = max_cover_sampling(node_graph, num_samples)
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
        samples, rem_nodes = max_cover_sampling(node_graph, num_samples)
        current_coverage = (total_num - rem_nodes) / total_num

        if current_coverage < coverage:
            sim_upper = sim
        else:
            sim_lower = sim
        sim = (sim_upper + sim_lower) / 2
    # print(f"Converged to tau = {sim/1000}")
    return sim / 1000, node_graph, samples

def max_k_cover(data, Ks, sim_thresh=0.7):
    cos_sim = cosine_similarity(data)
    selected_samples = {}
    for K in Ks:
        cap = (2 * 0.9 * len(data)) / K
        G = build_graph(cos_sim, sim_thresh=sim_thresh) #, max_degree=cap)
        selected_samples[K], _ = max_cover_sampling(G, K)
    return selected_samples

# METHODS TO IMPLEMENT

# Curriculum learning

# Pseudorandom Max K Cover

# Hierarchical Max K Cover