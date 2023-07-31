import math
from abc import ABC, abstractmethod
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class PlottingStrategy(ABC):
    # colors for each dataset in the plot
    colors: dict[str, str] = {}

    def plot(self, src, ref_path, ref_file):
        """
        Plot data

        TODO explain usage in detail
        """
        # read in data to plot
        history = pd.read_json(src, orient="split")
        # read in references
        ref_path = path.join(
            path.dirname(path.realpath(__file__)),
            ref_path,
            ref_file,
        )
        reference = pd.read_json(ref_path, orient="split")
        history = pd.concat([history, reference])

        history = self.make_history_adjustments(history)

        # define lines to plot
        lines = self.get_line_plotting_dict()

        seen_labels = []  # keep track to ensure proper legends

        # group plots by combinations of k and r
        groups = history.groupby(list(self.get_parameters_to_group_by()))
        # calculate the maximum value of the four relevant value columns
        max_val_in_graph = (
            history[self.get_plotted_column_name_list()].max(numeric_only=True).max()
        )
        # calculate the necessary exponent for the y-axis scaling
        y_scale_exponent = len(str(math.ceil(max_val_in_graph)))

        fig, axs = plt.subplots(1, len(groups), sharey=True)
        if len(groups) == 1:
            axs = [axs]
        for ax, (param_val_tuple, group) in zip(axs, groups):
            ax.set_title(
                convert_parameter_tuple_to_string(
                    self.get_parameters_to_group_by(), param_val_tuple
                )
            )
            ax.set_xlim(10**2, 10**4)
            ax.set_ylim(300, 10**y_scale_exponent)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(self.get_x_label())
            ax.set_ylabel(self.get_y_label())
            ax.grid(which="both")

            # group lines by implementation
            impls = group.groupby("impl")
            for name, impl in impls:
                means = self.calculate_means(impl)
                errors = self.calculate_errors(impl)
                for col, label in lines.items():
                    text = f"{label} ({name})"
                    if text in seen_labels:
                        text = "__nolabel__"
                    else:
                        seen_labels.append(text)

                    ax.plot(
                        means.index,
                        means[col],
                        self.get_plot_point_symbol(label),
                        label=text,
                        color=self.color_for_impl(name),
                    )
                    ax.fill_between(
                        means.index,
                        means[col] + errors[col],
                        means[col] - errors[col],
                        alpha=0.4,
                        color=self.color_for_impl(name),
                    )

        fig.legend(loc="upper center")
        plt.subplots_adjust(top=0.7)
        plt.show()

    @abstractmethod
    def get_x_label(self):
        return "default x label"

    @abstractmethod
    def get_y_label(self):
        return "default y label"

    @abstractmethod
    def get_parameters_to_group_by(self):
        return ()

    @abstractmethod
    def make_history_adjustments(self, history):
        return history

    @abstractmethod
    def get_line_plotting_dict(self):
        return {}

    @abstractmethod
    def get_plotted_column_name_list(self):
        return []

    @abstractmethod
    def calculate_means(self, impl):
        pass

    @abstractmethod
    def calculate_errors(self, impl):
        pass

    @abstractmethod
    def get_plot_point_symbol(self, label):
        return "o"

    def color_for_impl(self, impl):
        """
        Returns a color for a given key `impl`, and generates a new unique color if it does not exist.
        """

        if impl in self.colors:
            return self.colors[impl]

        mcolor_names: list = [
            c for c in mcolors.CSS4_COLORS if c not in self.colors.values()
        ]
        new_color = np.random.choice(mcolor_names)
        self.colors[impl] = new_color
        return new_color


def convert_parameter_tuple_to_string(name_tuple, value_tuple):
    string = ""
    for name, item in zip(name_tuple, value_tuple):
        if string != "":
            string = string + ", "
        string = string + f"{name} = {item}"
    return string
