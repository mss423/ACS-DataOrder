import matplotlib.pyplot as plt
import seaborn as sns

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

# Pass set of results for different models fit to a given order
def plot_results(metrics, normalization, trivial=1.0):
	fig, ax = plt.subplots(1,1)
	ax.axhline(trivia, ls="--", color="gray")
	
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

	legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
	fig.set_size_inches(4, 3)
	for line in legend.get_lines():
		line.set_linewidth(3)
