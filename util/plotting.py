import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import os.path as path

def plot(src, quantum_factor=2):
    colors = {
        "KIT": "green",
        "RUB": "blue",
        "Cade": "orange"
    }
    def color_for_impl(impl):
        """
            Returns a color given a key. Does not duplicate colors so it might run
            out of colors but who is going to print that much data :)
        """
        try:
            return colors[impl]
        except KeyError:
            mcolor_names: list = [c for c in mcolors.CSS4_COLORS.keys() if c not in colors.values()]
            new_color = random.choice(mcolor_names)
            colors[impl] = new_color
            return new_color

    # read in data to plot
    history = pd.read_json(src, orient="split")
    # read in references TODO: make this optional via additional arguments
    ref_path = path.join(path.dirname(path.realpath(__file__)), 
                         "../data/plot_reference/hill_climb_cade.json")
    reference = pd.read_json(ref_path, orient="split")
    history = pd.concat([history, reference])

    # compute combined query costs of quantum search
    c = history["quantum_expected_classical_queries"]
    q = history["quantum_expected_quantum_queries"]
    history["quantum_cqq"] = c + quantum_factor * q  

    # define lines to plot
    lines = {
        "classical_actual_queries": "Classical Queries",
        "quantum_cqq": "Quantum Queries",
    }
    seen_labels = [] # keep track to ensure proper legends

    # group plots by combinations of k and r
    groups = history.groupby(["k", "r"])
    fig, axs = plt.subplots(1, len(groups), sharey=True)
    if len(groups) == 1:
        axs = [axs]
    for ax, ((k, r), group) in zip(axs, groups):
        ax.set_title(f"k = {k}, r = {r}")
        ax.set_xlim(10**2, 10**4)
        ax.set_ylim(300, 10**5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_ylabel("Queries")
        ax.grid(which="both")

        # group lines by implementation
        impls = group.groupby("impl")
        for name, impl in impls:
            means = impl.groupby("n").mean(numeric_only=True)
            errors = impl.groupby("n").sem(numeric_only=True)
            for col, label in lines.items():

                text = f"{label} ({name})"
                if text in seen_labels:
                    text = "__nolabel__"
                else:
                    seen_labels.append(text)

                ax.plot(
                    means.index,
                    means[col], 
                    "x" if "Quantum" in label else "o", 
                    label=text, 
                    color=color_for_impl(name))
                ax.fill_between(
                    means.index,
                    means[col] + errors[col],
                    means[col] - errors[col],
                    alpha=0.4,
                    color=color_for_impl(name)
                )
        

    fig.legend(loc="upper center")
    plt.subplots_adjust(top=0.7)
    plt.show()
