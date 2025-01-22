import networkx as nx
import numpy as np
import random

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
			if similarity >= sim_thresh:
				G.add_edge(i, j, weight=similarity)
		# add self-loop, doesn't count toward max_degree
		G.add_edge(i, i, weight=1)
	return G

def build_dist_graph(data, threshold=0.5, alpha=1.0):
	G = nx.Graph()
	G.add_nodes_from(range(data.shape[0]))  # Add nodes

	distances = pdist(data, metric='euclidean') 
	
	# Applying sigmoid function
	normalized_distances = 1 / (1 + np.exp(alpha * distances))  

	# Get indices of points where normalized distance is above threshold (since it's now similarity)
	indices_i, indices_j = np.where(squareform(normalized_distances) > threshold)

	# Add edges to the graph
	edges = list(zip(indices_i, indices_j))
	G.add_edges_from(edges)

	return G

def recursive_build_graph(cos_sim, sim_thresh=0.0, max_degree=None, labels=None, covered=None):
	G = nx.Graph()
	for i in range(len(cos_sim)):
		G.add_node(i)
		if covered and i in covered:
			G.add_edge(i,i, weight=1)
			continue
		# Sort neighbors by similarity in descending order
		neighbors = sorted(enumerate(cos_sim[i]), key=lambda x: x[1], reverse=True)
		for j, similarity in neighbors:
			if j == i:
				continue
			if max_degree and G.degree(i) >= max_degree:
				break  # Exit the inner loop if max_degree is reached
			if similarity >= sim_thresh:
				G.add_edge(i, j, weight=similarity)
		# add self-loop, doesn't count toward max_degree
		G.add_edge(i, i, weight=1)
	return G

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
		samples, rem_nodes = max_cover(node_graph, num_samples)
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
		samples, rem_nodes = max_cover(node_graph, num_samples)
		current_coverage = (total_num - rem_nodes) / total_num

		if current_coverage < coverage:
			sim_upper = sim
		else:
			sim_lower = sim
		sim = (sim_upper + sim_lower) / 2
	# print(f"Converged to tau = {sim/1000}")
	return sim / 1000, node_graph, samples

def binary_thresh_search(data, num_samples, coverage, cap=None, epsilon=None, sims=[0,1000], labels=None, covered=None):
	total_num = len(data)
	if epsilon is None:
		# There is a chance that we never get close enough to "coverage" to terminate
		# the loop. I think at the very least we should have epsilon > 1/total_num.
		# So let's set set epsilon equal to the twice of the minimum possible change
		# in coverage.
		epsilon = 5 * 10 / total_num  # Dynamic epsilon

	if coverage < num_samples / total_num:
		node_graph = build_graph(data, 1)
		samples, rem_nodes = max_cover(node_graph, num_samples)
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
	cap = (2 * total_num * coverage) / num_samples
	while abs(current_coverage - coverage) > epsilon and sim_upper - sim_lower > 1:
		if count >= max_run:
			print(f"Reached max number of iterations ({max_run}). Breaking...")
			break
		count += 1

		node_graph = build_graph(data, sim / 1000, max_degree=cap, labels=labels)
		samples, rem_nodes = max_cover_recursive(node_graph, num_samples, covered=covered)
		current_coverage = (total_num - rem_nodes) / total_num

		if current_coverage < coverage:
			sim_upper = sim
		else:
			sim_lower = sim
		sim = (sim_upper + sim_lower) / 2
		# print(f"Converged to tau = {sim/1000}")
	covered = samples
	return sim / 1000, node_graph, samples

def max_cover(graph, k):
	nodes = list(graph.nodes())
	selected_nodes = []
	covered_nodes = set()

	for _ in range(k):
		if not nodes:
			break
		max_cover_node = max([node for node in nodes if node not in covered_nodes],
			key=lambda n: len([neighbor for neighbor in graph.neighbors(n) if neighbor not in covered_nodes])
			)
		selected_nodes.append(max_cover_node)
		covered_nodes.add(max_cover_node)
		covered_nodes.update(graph.neighbors(max_cover_node))

		# Remove neighbors of selected node
		for neighbor in graph.neighbors(max_cover_node):
			if neighbor in nodes:
				nodes.remove(neighbor)
	return selected_nodes, len(nodes)

def max_cover_recursive(graph, k, covered=None):
	nodes = list(graph.nodes())
	selected_nodes = []
	covered_nodes = set()
	if covered:
		for x in covered:
			covered_nodes.add(nodes[x])

	for _ in range(k):
		if not nodes:
			break
		# print(len(covered_nodes))
		max_cover_node = max([node for node in nodes if node not in covered_nodes],
			key=lambda n: len([neighbor for neighbor in graph.neighbors(n) if neighbor not in covered_nodes])
			)
		selected_nodes.append(max_cover_node)
		covered_nodes.add(max_cover_node)
		covered_nodes.update(graph.neighbors(max_cover_node))

		# Remove neighbors of selected node
		for neighbor in graph.neighbors(max_cover_node):
			if neighbor in nodes:
				nodes.remove(neighbor)
	return selected_nodes, len(nodes)

def max_cover_cluster(graph, k):
	nodes = list(graph.nodes())
	selected_nodes = []
	covered_nodes = set()
	cluster_assignments = {}

	for i in range(k):
		if not nodes:
			break
		max_cover_node = max([node for node in nodes if node not in covered_nodes],
			key=lambda n: len([neighbor for neighbor in graph.neighbors(n) if neighbor not in covered_nodes])
			)
		selected_nodes.append(max_cover_node)
		covered_nodes.add(max_cover_node)
		# covered_nodes.update(graph.neighbors(max_cover_node))

		# Assign clusters
		cluster_assignments[max_cover_node] = i
		for neighbor in graph.neighbors(max_cover_node):
			if neighbor not in covered_nodes:
				cluster_assignments[neighbor] = i
				covered_nodes.add(neighbor)

		# Remove neighbors of selected node
		for neighbor in graph.neighbors(max_cover_node):
			if neighbor in nodes:
				nodes.remove(neighbor)
	return selected_nodes, len(nodes), cluster_assignments

def max_cover_sampling(graph, K):
	"""
	Performs max cover on the graph, extracts K cover points, 
	and samples the remaining points into K clusters.

	Args:
	graph: NetworkX graph object representing the data points.
	K: Number of clusters to create.

	Returns:
	A list of lists, where each sublist contains the indices of the 
	data points belonging to each cluster.
	"""

	# 1. Max Cover
	cover_points, _, cluster_assignments = max_cover_cluster(graph, K)  # Implement your max_cover function here
	num_clusters = len(cover_points)

	# 2. Initialize clusters
	clusters = [[] for _ in range(num_clusters)]
	for i in range(num_clusters):
		clusters[i].append(cover_points[i])
		for node, cluster_id in cluster_assignments.items():
			if cluster_id == i:
				clusters[i].append(node)

	# 3. Calculate cluster probabilities
	cluster_sizes = [len(cluster) for cluster in clusters]
	total_size = sum(cluster_sizes)
	cluster_probs = [size / total_size for size in cluster_sizes]

	# 4. Repeatedly sample without replacement
	sampled_indices = []
	# sampled_indices = cover_points # start with max cover cluster centers
	while len(sampled_indices) < len(graph.nodes):
		# Select a cluster based on probabilities
		selected_cluster = np.random.choice(num_clusters, p=cluster_probs)
		# Sample a point from the selected cluster
		sampled_index = random.choice(clusters[selected_cluster])
		# Check if already sampled
		if sampled_index not in sampled_indices:
			sampled_indices.append(sampled_index)
			# Remove sampled point from the cluster
			clusters[selected_cluster].remove(sampled_index)
			# Update cluster sizes and probabilities
			cluster_sizes[selected_cluster] -= 1
			total_size -= 1
			cluster_probs = [size / total_size for size in cluster_sizes]

	return sampled_indices