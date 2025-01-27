from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from acs import *
from universal_order_scratch import *

import matplotlib.pyplot as plt

def max_cover_order(data, threshold=0.5):
    G = create_graph(data, threshold)
    _, node_order = run_max_cover(G)

    all_idx = set(list(range(len(data))))
    remaining_indices = list(all_idx - set(node_order))
    return node_order + remaining_indices

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
    # 1) Build the hierarchy
    root = hierarchical_acs_tree(data, coverage=0.9)

    # 2) Get a total ordering of original indices
    return get_total_ordering(root)

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
        n_clusters = data.shape[1]
    
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

# ---------------------- #

def get_order(data, method_name):
    name_to_fn = {
        "max_cover": max_cover_order, #max_cover_random,
        "pseudo": max_cover_pseudo,
        "acs": acs_k_cover,
        "hier_max": total_order,
        "kmeans": kmeans_order,
        "hier_acs": hierarchical_acs
    }

    if method_name not in name_to_fn:
        print("Unknown ordering method!")
        raise NotImplementedError

    order_fn = name_to_fn[method_name]
    order = []
    for i in range(data.shape[0]):
        cur_batch = np.array(data[i])
        if method_name == "hier_max":
            hierarchy = hierarchical_max_cover(cur_batch)
            order.append(order_fn(hierarchy))
            continue
        elif method_name == "acs":
            order.append(order_fn(cur_batch, cur_batch.shape[1]))
            continue
        order.append(order_fn(cur_batch))
    order = torch.tensor(order, dtype=torch.int64)
    return order[:, :, None]


# ----------------------  DEFUNCT ---------------------- #

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

# def hierarchical_acs(data):
#     selected_samples = []
#     K = len(data) // 2
#     cos_sim = cosine_similarity(data)

#     while K >= 1 and len(selected_samples) != len(data):
#         _, _, samples = calculate_similarity_threshold(cos_sim, K, coverage=1.0, sims=[0,1000])
#         for s in samples:
#             if s not in selected_samples:
#                 selected_samples.append(s)

#         K = K // 2

#     if len(selected_samples) < len(data):
#         all_idx = set(list(range(len(data))))
#         remaining_indices = list(all_idx - set(selected_samples))
#         return selected_samples + remaining_indices

#     return selected_samples
