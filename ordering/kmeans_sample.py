from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd

def kmeans_sample(data_df, K):
	embed_data = get_embeddings_task(data_df['sentence'])

	# Extract embedding data
	# embed_data = np.array(vertex_embeddings)
	
	# Find kmeans centers
	kmeans = KMeans(n_clusters=K).fit(embed_data)

	# Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
	closest_pt_idx = []
	for iclust in range(kmeans.n_clusters):
		# get all points assigned to each cluster:
		cluster_pts = embed_data[kmeans.labels_ == iclust]

		# get all indices of points assigned to this cluster:
		cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]

		cluster_cen = kmeans.cluster_centers_[iclust]
		min_idx = np.argmin([euclidean(embed_data[idx], cluster_cen) for idx in cluster_pts_indices])
		closest_pt_idx.append(cluster_pts_indices[min_idx])

	return data_df.iloc[closest_pt_idx]