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

def max_cover_random(data, sim_thresh=0.7, seed=42):
    # Runs max cover on graph with similarity threshold, then randomly permutes remaining data
    np.random.seed(seed)
    cos_sim = cosine_similarity(data)
    G = build_graph = build_graph(cos_sim, sim_thresh=sim_thresh)
    samples, _ max_cover(G, len(data))

    all_idx = set(list(range(len(data))))
    remaining_indices = list(all_idx - set(samples))
    # Return the max cover and randomly permuted remainder
    return samples, np.random.shuffle(remaining_indices)

def max_cover_pseudo(data, sim_thresh=0.7, seed=42):
    # Runs max cover on graph with similarity threshold, then randomly permutes remaining data
    np.random.seed(seed)
    cos_sim = cosine_similarity(data)
    G = build_graph = build_graph(cos_sim, sim_thresh=sim_thresh)
    samples, _ = max_cover_sampling(G, len(data))

    return samples

def max_k_cover(data, Ks, sim_thresh=0.7):
    # Return K cover for a given similarity threshold
    # Note: yields <= K samples
    cos_sim = cosine_similarity(data)
    selected_samples = {}
    for K in Ks:
        G = build_graph(cos_sim, sim_thresh=sim_thresh)
        selected_samples[K], _ = max_cover(G, K)
    return selected_samples

def acs_k_cover(data, K):
    # For fixed K coverage, compute optimal threshold and return K samples
    cos_sim = cosine_similarity(data)
    _, _, samples = calculate_similarity_threshold(cos_sim, K, coverage=0.9)
    return samples

def acs_ks_range(data, Ks):
    # Iterate over each K, run ACS to get samples
    # Note: this ordering is NOT nested
    cos_sim = cosine_similarity(data)
    selected_samples = {}
    for K in Ks:
        _, _, selected_samples[K] = calculate_similarity_threshold(cos_sim, K, coverage=0.9)
    return selected_samples

# METHODS TO IMPLEMENT

# Curriculum learning

# Hierarchical Max K Cover