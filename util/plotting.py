import pandas as pd
import matplotlib.pyplot as plt

def plot(src):
    history = pd.read_json(src, orient="split")

    # compute combined query costs of quantum search
    c = history["quantum_search_expected_classical_queries"]
    q = history["quantum_search_expected_quantum_queries"]
    # history["quantum_search_cq"] = c + q
    history["quantum_search_cqq"] = c + 2 * q
    # history["quantum_search_qq"] = 2 * q

    # plot
    lines = {
        "classical_search_actual_queries": "classical search (actual)",
        # "classical_search_expected_queries": "classical search (expected)",
        # "quantum_search_cq": "quantum search (expected classical + quantum)",
        "quantum_search_cqq": "quantum search (expected classical + 2 quantum)",
        # "quantum_search_qq": "quantum search (2 quantum)",
    }
    groups = history.groupby(["k", "r"])
    fig, axs = plt.subplots(1, len(groups), sharey=True)
    if len(groups) == 1:
        axs = [axs]
    for ax, ((k, r), group) in zip(axs, groups):
        means = group.groupby("n").mean()
        errors = group.groupby("n").sem()
        ax.set_title(f"k = {k}, r = {r}")
        ax.set_xlim(10**2, 10**4)
        ax.set_ylim(300, 10**5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_ylabel("Queries")
        ax.grid(which="both")
        first = ax == axs[0]
        for col, label in lines.items():
            ax.plot(means.index, means[col], "x" if "quantum" in label else "o", label=label if first else None, color="b")
            ax.fill_between(
                means.index,
                means[col] + errors[col],
                means[col] - errors[col],
                alpha=0.5,
                color="b"
            )
        
        # Default data
        # Comparative parameters from Cade et al.
        ax.plot((100, 300, 1000, 3000, 10000), (400, 1.9e3, 7.5e3, 2.8e4, 1e5),
            **{'color': 'orange', 'marker': 'o', 'label': 'Cade et al. (Classical)'})
        ax.plot((100, 300, 1000, 3000, 10000), (2e3, 4.5e3, 1.2e4, 3.0e4, 8e4), 
            **{'color': 'orange', 'marker': 'x', 'label': 'Cade et al. (Quantum)'})

    fig.legend(loc="upper center")
    plt.subplots_adjust(top=0.7)
    plt.show()