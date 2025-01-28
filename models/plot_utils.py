import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name

def order_names(name):
    if "random" in name:
        return "Random"
    if "hier_max" in name:
        return "Hierarchical Max Coverage"
    if "hier_acs" in name:
        return "Hierarchical ACS"
    if "pseudo" in name:
    	return "Pseudorandom"
    if "acs" in name:
    	return "ACS"
    if "max_cover" in name:
    	match = re.search(r"max_cover_k=([\d.]+)", method_name)
    	tau = float(match.groupd(1))
    	return "Max Coverage, K = " + f"{tau}"
    if "kmeans" in name:
    	return "k Means"

# Pass set of results for different models fit to a given order
def plot_results(metrics, normalization, trivial=1.0, xlim=None, ylim=None):
	fig, ax = plt.subplots(1,1)
	ax.axhline(trivial, ls="--", color="gray")
	
	color = 0
	for name, vs in metrics.items():
		m_processed = {}
		for k,v in vs.items():
			v = [vv / normalization for vv in v]
			m_processed[k] = v

		ax.plot(m_processed["mean"], "-", label=baseline_names(name), color=palette[color % 10], lw=2)
		low = m_processed["bootstrap_low"]
		high = m_processed["bootstrap_high"]
		ax.fill_between(range(len(low)), low, high, alpha=0.3)
		color += 1
	ax.set_xlabel("in-context examples")
	ax.set_ylabel("squared error")
	ax.set_xlim(-1, 40)
	ax.set_ylim(-0.1, 1.25)
	if xlim:
		ax.set_xlim(xlim[0], xlim[1])
	if ylim:
		ax.set_ylim(ylim[0], ylim[1])

	legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
	fig.set_size_inches(4, 3)
	for line in legend.get_lines():
		line.set_linewidth(3)

def plot_results_model(results, normalization, model, trivial=1.0, xlim=None, ylim=None, opt=None):
	fig, ax = plt.subplots(1,1)
	ax.axhline(trivial, ls="--", color="gray")
	
	color = 0
	for order in results.keys():
		metrics = results[order]
		for name, vs in metrics.items():
			if name != model:
				continue
			m_processed = {}
			for k,v in vs.items():
				v = [vv / normalization for vv in v]
				m_processed[k] = v
			ax.plot(m_processed["mean"], "-", label=order_names(order), color=palette[color % 10], lw=2)
			low = m_processed["bootstrap_low"]
			high = m_processed["bootstrap_high"]
			ax.fill_between(range(len(low)), low, high, alpha=0.3)
			color += 1

	ax.set_xlabel("# of Samples")
	ax.set_ylabel("Squared Error")
	ax.set_xlim(-1, 40)
	ax.set_ylim(-0.1, 1.25)
	if xlim:
		ax.set_xlim(xlim[0], xlim[1])
	if ylim:
		ax.set_ylim(ylim[0], ylim[1])
	if opt:
		ax.axvline(opt, ls="--", color="red")

	# legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
	legend = ax.legend(loc="upper right", bbox_to_anchor=(1, 1)) # Modified line
	fig.set_size_inches(8, 6)
	for line in legend.get_lines():
		line.set_linewidth(3)

def plot_results_baseline(results, model, xlim=None, ylim=None):
	fig, ax = plt.subplots(1,1)
	ax.axhline(0.0, ls="--", color="gray")

	baseline = results["random"]
	
	color = 0
	for order in results.keys():
		if order == "random":
			continue
		metrics = results[order]
		for name, vs in metrics.items():
			if name != model:
				continue
			m_processed = {}
			for k,v in vs.items():
				v = [vv / b for vv, b in zip(v, baseline[name][k])]
				m_processed[k] = v
			ax.plot([x - 1 for x in m_processed["mean"]], "-", label=order_names(order), color=palette[color % 10], lw=2)
			low = [x - 1 for x in m_processed["bootstrap_low"]]
			high = [x - 1 for x in m_processed["bootstrap_high"]]
			ax.fill_between(range(len(low)), low, high, alpha=0.3)
			color += 1

	ax.set_xlabel("# of Samples")
	ax.set_ylabel("% Improvement over Random")
	ax.set_xlim(-1, 40)
	ax.set_ylim(-1, 1)
	if xlim:
		ax.set_xlim(xlim[0], xlim[1])
	if ylim:
		ax.set_ylim(ylim[0], ylim[1])

	# legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
	legend = ax.legend(loc="upper right", bbox_to_anchor=(1, 1)) # Modified line
	fig.set_size_inches(8, 6)
	for line in legend.get_lines():
		line.set_linewidth(3)
