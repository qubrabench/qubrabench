import math
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class PlottingStrategy(ABC):
    """
    Generic plotting strategy to visualize benchmarking data.

    Assumes data is given as a pandas dataframe.
    See the abstract methods below for configuration options.
    """

    # colors for each dataset in the plot
    colors: dict[str, str] = {}

    @abstractmethod
    def x_axis_column(self) -> str:
        """
        Data column to plot along the X-axis
        """
        return ""

    @abstractmethod
    def x_axis_label(self) -> str:
        """
        Label to display for the X-axis
        """
        return ""

    @abstractmethod
    def y_axis_label(self) -> str:
        """
        Label to display for the Y-axis
        """
        return ""

    @abstractmethod
    def get_plot_group_column_names(self) -> list[str]:
        """
        Generate a plot for each unique tuple of values for the specified columns.
        Example: ["k", "n"] - a plot will be generated for each unique tuple value (k, n).
        Example: [] - generate a single plot with the entire data.

        Returns:
            List of column names to group by.
        """
        return []

    @abstractmethod
    def get_data_group_column_names(self) -> list[str]:
        """
        Generate a data line for each unique value in the specified columns.
        Useful if you the data was generated with different tags based on implementation source, parameter choice etc., that one wants to compare against in a single plot.

        Example: ["impl"] - a line will be generated for each unique `impl` label.
        """

    @abstractmethod
    def compute_aggregates(
        self, data: pd.DataFrame, *, quantum_factor: float
    ) -> pd.DataFrame:
        return data

    @abstractmethod
    def get_column_names_to_plot(self) -> dict[str, tuple[str, str]]:
        """
        Dictionary of columns to display in the plot.
            Key: is the column name in the dataframe
            Value: Column display name, Marker to use in the plot

        Example:
            {"c": ("Classical", "o"), "q": ("Quantum", "x")}
        """
        return {}

    def plot(self, data: pd.DataFrame, *, quantum_factor: float = 2):
        """
        Plot benchmarking data.

        Args:
            data: a pandas DataFrame containing all the benchmark data.
            quantum_factor: conversion factor for the cost of a quantum query (w.r.t. classical queries).
        """

        data = self.compute_aggregates(data, quantum_factor=2)

        seen_labels = []  # keep track to ensure proper legends

        # calculate the maximum value of the four relevant value columns
        max_val_in_graph = (
            data[list(self.get_column_names_to_plot().keys())]
            .max(numeric_only=True)
            .max()
        )
        # calculate the necessary exponent for the y-axis scaling
        y_scale_exponent = len(str(math.ceil(max_val_in_graph)))

        # make groups to generate plots for
        groups = data.groupby(self.get_plot_group_column_names())

        fig, axs = plt.subplots(1, len(groups), sharey=True)
        if len(groups) == 1:
            axs = [axs]

        for ax, (plot_params, group) in zip(axs, groups):
            ax.set_title(self.make_plot_title(plot_params))

            ax.set_xlim(10**2, 10**4)
            ax.set_ylim(300, 10**y_scale_exponent)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(self.x_axis_label())
            ax.set_ylabel(self.y_axis_label())
            ax.grid(which="both")

            # group lines by implementation
            impls = group.groupby(self.get_data_group_column_names())
            for impl_params, impl in impls:
                plot_data = impl.groupby(self.x_axis_column())
                means = plot_data.mean(numeric_only=True)
                errors = plot_data.sem(numeric_only=True)

                impl_name = self.make_plot_label(impl_params)
                for col, (col_name, marker) in self.get_column_names_to_plot().items():
                    text = f"{col_name} ({impl_name})"
                    if text in seen_labels:
                        text = "__nolabel__"
                    else:
                        seen_labels.append(text)

                    ax.plot(
                        means.index,
                        means[col],
                        marker,
                        label=text,
                        color=self.color_for_data_group(impl_name),
                    )
                    ax.fill_between(
                        means.index,
                        means[col] + errors[col],
                        means[col] - errors[col],
                        alpha=0.4,
                        color=self.color_for_data_group(impl_name),
                    )

        fig.legend(loc="upper center")
        plt.subplots_adjust(top=0.7)
        plt.show()

    def make_plot_title(self, plot_params: list) -> str:
        """
        Generate the heading of each plot from the list of values in columns ``self.get_plot_group_column_names()``
        """
        return self.serialize_value_tuple(
            self.get_plot_group_column_names(), plot_params
        )

    def color_for_data_group(self, data_group: str):
        """
        data_group is a tuple of values in columns ``self.get_data_group_column_names()``

        Returns:
             a color for a given key, and generates a new unique color if it does not exist.
        """

        if data_group in self.colors:
            return self.colors[data_group]

        mcolor_names: list = [
            c for c in mcolors.CSS4_COLORS if c not in self.colors.values()
        ]
        new_color = np.random.choice(mcolor_names)
        self.colors[data_group] = new_color
        return new_color

    def make_plot_label(self, data_params: list) -> str:
        """
        Generate the label for each line from the list of values in columns `self.get_data_group_column_names()`
        """
        return self.serialize_value_tuple(
            self.get_data_group_column_names(), data_params
        )

    @staticmethod
    def serialize_value_tuple(columns: list[str], values: list[str]) -> str:
        """
        Serialize a set of column values from a given table row to display.

        Args:
            columns: column header names
            values: column values in the row of interest
        """
        columns = [f"{column} = {value}" for (column, value) in zip(columns, values)]
        return ", ".join(columns)
