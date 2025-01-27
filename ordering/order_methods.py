from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
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

# ---------------------- #

def get_order(data, method_name):
    name_to_fn = {
        "max_cover": max_cover_order, #max_cover_random,
        "pseudo": max_cover_pseudo,
        "acs": acs_k_cover,
        "hier_max": build_total_order,
        "kmeans": kmeans_order
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
            thresholds = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            hierarchy = hierarchical_max_cover(cur_batch, thresholds, verbose=False)
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
