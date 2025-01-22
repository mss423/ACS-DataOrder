from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import *

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

def max_cover_random(data, threshold=0.0, seed=42, max_degree=None):
    # Runs max cover on graph with similarity threshold, then randomly permutes remaining data
    np.random.seed(seed)

    cos_sim = cosine_similarity(data)
    cos_sim = np.clip(cos_sim, -1, 1)

    G = build_graph(cos_sim, sim_thresh=threshold, max_degree=max_degree)
    samples, _ = max_cover(G, len(data))

    all_idx = set(list(range(len(data))))
    remaining_indices = list(all_idx - set(samples))
    # Return the max cover and randomly permuted remainder
    return samples, remaining_indices

def max_cover_pseudo(data, threshold=0.0, seed=42, max_degree=None):
    # Runs max cover on graph with similarity threshold, then randomly permutes remaining data
    np.random.seed(seed)

    cos_sim = cosine_similarity(data)
    cos_sim = np.clip(cos_sim, -1, 1)

    G = build_graph(cos_sim, sim_thresh=threshold, max_degree=max_degree)
    
    samples = max_cover_sampling(G, len(data))
    return samples

def acs_k_cover(data, K):
    # For fixed K coverage, compute optimal threshold and return K samples
    cos_sim = cosine_similarity(data)
    cos_sim = np.clip(cos_sim, -1, 1)

    _, _, samples = calculate_similarity_threshold(scaled_sim, K, coverage=0.9, sims=[-1000,1000])
    return samples

def hierarchical_acs(data, covered=None):
    if covered and len(covered) == 1:
        return []
    cos_sim = cosine_similarity(data)
    if covered: 
        K = len(covered) // 2
    else:
        K = len(data) // 2
    thresh, _, selected_samples = binary_thresh_search(cos_sim, K, coverage=0.9, sims=[0,1000], covered=covered)
    print(f"Sim_thresh = {thresh}, k = {len(selected_samples)}")
    return selected_samples + hierarchical_acs(data, selected_samples)

def hierarchical_max_cover(data, initial_threshold=0.9, threshold_step=0.1):
    """
    Performs hierarchical max cover with decreasing similarity thresholds.

    Args:
        data (np.ndarray): The input data.
        initial_threshold (float, optional): The initial similarity threshold. Defaults to 0.9.
        threshold_step (float, optional): The step size for decreasing the threshold. Defaults to 0.1.

    Returns:
        list: A list of indices representing the data ordering.
    """

    all_samples = list(range(len(data)))  # Initialize with all data point indices
    covered_samples = []  # Initialize covered samples as an empty list
    threshold = initial_threshold
    
    while len(covered_samples) < len(all_samples):
        # Construct the similarity matrix
        cos_sim = cosine_similarity(data)
        
        # Build the graph for the current threshold
        node_graph = build_graph(data, threshold, max_degree=len(data)) # No cap on degree for max cover

        # Find the max cover set for uncovered points
        uncovered_samples = list(set(all_samples) - set(covered_samples))
        
        # If no uncovered samples are left, break out of loop
        if not uncovered_samples:
            break

        # Map uncovered samples to original indices within the graph
        uncovered_indices_in_graph = [all_samples.index(sample) for sample in uncovered_samples]

        # Create a subgraph containing only uncovered samples
        subgraph = {node: [neighbor for neighbor in neighbors if neighbor in uncovered_indices_in_graph] 
                   for node, neighbors in node_graph.items() if node in uncovered_indices_in_graph}

        # Select points for current similarity threshold using max cover
        selected_samples_indices, _ = max_cover(subgraph, len(uncovered_samples))  # select all remaining points
        
        # Map selected indices back to original indices within all samples
        selected_samples = [uncovered_samples[idx] for idx in selected_samples_indices]
        
        # Add selected samples to covered samples
        covered_samples.extend(selected_samples)

        # Decrease the similarity threshold
        threshold -= threshold_step
        threshold = max(0, threshold) # Avoid negative thresholds

    return covered_samples


# METHODS TO IMPLEMENT

# Curriculum learning (?)

# Hierarchical Max K Cover