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

    @abstractmethod
    def x_axis_label(self) -> str:
        """
        Label to display for the X-axis
        """

    @abstractmethod
    def y_axis_label(self) -> str:
        """
        Label to display for the Y-axis
        """

    @abstractmethod
    def get_plot_group_column_names(self) -> list[str]:
        """
        Generate a plot for each unique tuple of values for the specified columns.
        Example: ["k", "n"] - a plot will be generated for each unique tuple value (k, n).
        Example: [] - generate a single plot with the entire data.

        Returns:
            List of column names to group by.
        """

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
        """
        Compute any additional data columns needed for plotting

        Args:
            data: a pandas DataFrame with the input benchmark data
            quantum_factor: the conversion cost factor for quantum queries (w.r.t. classical queries)
        """

    @abstractmethod
    def get_column_names_to_plot(self) -> dict[str, tuple[str, str]]:
        """
        Dictionary of columns to display in the plot.
            Key: is the column name in the dataframe
            Value: Column display name, Marker to use in the plot

        Example:
            {"c": ("Classical", "o"), "q": ("Quantum", "x")}
        """

    def plot(
        self, data: pd.DataFrame, *, quantum_factor: float = 2, y_lower_lim: float = 1
    ):
        """
        Plot benchmarking data.

        Args:
            data: a pandas DataFrame containing all the benchmark data.
            quantum_factor: conversion factor for the cost of a quantum query (w.r.t. classical queries).
            y_lower_lim: lower limit on the Y-axis (useful if the data starts at a large value)

        Raises:
            ValueError: if no columns are given to plot
        """

        if not self.get_column_names_to_plot():
            raise ValueError("no columns given to plot")

        data = self.compute_aggregates(data, quantum_factor=2)

        seen_labels = []  # keep track to ensure proper legends

        # calculate the range of the X-axis
        x_axis_data = data[self.x_axis_column()]
        x_min = x_axis_data.min(numeric_only=True).min()
        x_max = x_axis_data.max(numeric_only=True).max()

        # calculate the maximum value of the columns to be plotted, and the scaling on the Y-axis
        y_max = (
            data[list(self.get_column_names_to_plot().keys())]
            .max(numeric_only=True)
            .max()
        )
        y_scale_exponent = np.ceil(np.log10(y_max))

        plot_group_column_names = self.get_plot_group_column_names()
        data_group_column_names = self.get_data_group_column_names()

        # make groups to generate plots for
        if not plot_group_column_names:
            groups = [([], data)]
        else:
            groups = data.groupby(plot_group_column_names)

        fig, axs = plt.subplots(1, len(groups), sharey=True)
        if len(groups) == 1:
            axs = [axs]

        for ax, (plot_params, group) in zip(axs, groups):
            ax.set_title(self.make_plot_title(plot_params))

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_lower_lim, 10**y_scale_exponent)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(self.x_axis_label())
            ax.set_ylabel(self.y_axis_label())
            ax.grid(which="both")

            # group data lines
            if not data_group_column_names:
                impls = [([], group)]
            else:
                impls = group.groupby(data_group_column_names)

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


class BasicPlottingStrategy(PlottingStrategy, ABC):
    def __init__(self):
        self.colors[""] = "blue"

    def get_plot_group_column_names(self) -> list[str]:
        return []

    def get_data_group_column_names(self) -> list[str]:
        return []
