from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from universal_order_scratch import *

import matplotlib.pyplot as plt

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
    return samples + remaining_indices

def max_cover_pseudo(data, threshold=0.5, seed=42, max_degree=None):
    # Runs max cover on graph with similarity threshold, then randomly permutes remaining data
    np.random.seed(seed)

    cos_sim = cosine_similarity(data)
    cos_sim = np.clip(cos_sim, -1, 1)

    G = build_graph(cos_sim, sim_thresh=threshold, max_degree=max_degree)
    
    samples = max_cover_sampling(G, len(data))
    return samples

def acs_k_cover(data, K=None):
    if K == None:
        K = len(data) // 2
    # For fixed K coverage, compute optimal threshold and return K samples
    cos_sim = cosine_similarity(data)
    cos_sim = np.clip(cos_sim, -1, 1)

    _, _, samples = calculate_similarity_threshold(cos_sim, K, coverage=1.0, sims=[0,1000])
    all_idx = set(list(range(len(data))))
    remaining_indices = list(all_idx - set(samples))
    return samples + remaining_indices

def hierarchical_acs(data):
    selected_samples = []
    K = len(data) // 2
    cos_sim = cosine_similarity(data)

    while K >= 1 and len(selected_samples) != len(data):
        _, _, samples = calculate_similarity_threshold(cos_sim, K, coverage=1.0, sims=[0,1000])
        for s in samples:
            if s not in selected_samples:
                selected_samples.append(s)

        K = K // 2

    if len(selected_samples) < len(data):
        all_idx = set(list(range(len(data))))
        remaining_indices = list(all_idx - set(selected_samples))
        return selected_samples + remaining_indices

    return selected_samples

# def hierarchical_max_cover(data, initial_threshold=0.5, threshold_step=0.1):
    """
    Performs hierarchical max cover with decreasing similarity thresholds.

    Args:
        data (np.ndarray): The input data.
        initial_threshold (float, optional): The initial similarity threshold. Defaults to 0.9.
        threshold_step (float, optional): The step size for decreasing the threshold. Defaults to 0.1.

    Returns:
        list: A list of indices representing the data ordering.
    """

    # selected_samples = []  # Initialize covered samples as an empty list
    # threshold = initial_threshold
    # cos_sim = cosine_similarity(data)
    # cos_sim = np.clip(cos_sim, -1, 1)
    
    # while threshold >= 0.0 and len(selected_samples) != len(data):
    #     # Build the graph for the current threshold
    #     node_graph = build_graph(cos_sim, threshold) # No cap on degree for max cover
    #     samples, _ = max_cover(node_graph, len(data))
    #     # samples, _ = max_cover_debug(node_graph, len(data))

    #     for s in samples:
    #         if s not in selected_samples:
    #             selected_samples.append(s)

    #     # Decrease the similarity threshold
    #     threshold -= threshold_step

    # if len(selected_samples) < len(data):
    #     all_idx = set(list(range(len(data))))
    #     remaining_indices = list(all_idx - set(selected_samples))
    #     return selected_samples + remaining_indices

    # return selected_samples


# ---------------------- #

def get_order(data, method_name):
    name_to_fn = {
        "max_cover": max_cover_random,
        "pseudo": max_cover_pseudo,
        "acs": acs_k_cover,
        "hier_max": build_total_order,
        #"hier_acs": hierarchical_acs,
        #"hier_max1": hierarchical_flatten,
        #"hier_max2": alternative_2_ordering_all_data
    }

    if method_name not in name_to_fn:
        print("Unknown ordering method!")
        raise NotImplementedError

    order_fn = name_to_fn[method_name]
    order = []
    for i in range(data.shape[0]):
        cur_batch = np.array(data[i])
        if method_name == "hier_max":
            thresholds = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            hierarchy = hierarchical_max_cover(data, thresholds, verbose=False)
            order.append(order_fn(hierarchy))
            continue
        elif method_name == "acs":
            print(cur_batch.shape)
            order.append(order_fn(cur_batch, cur_batch.shape[1] * 2))
            continue
        order.append(order_fn(cur_batch))
    order = torch.tensor(order, dtype=torch.int64)
    return order[:, :, None]

