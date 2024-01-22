#!/usr/bin/env python
import networkx as nx

from graph_instances import random_fcs_graph
from graph_instances import random_lfr_graph
from dataclasses import asdict
import numpy as np
import pandas as pd
import qlouvain
from pathlib import Path
import os
import plotly.express as px
import plotly.graph_objects as go


# benchmarking
rng = np.random.default_rng()
dest = Path("edgeq.json")
instance_num = 3
graph_sizes = [250, 500, 1000, 2000, 5000, 10000]


def bench():
    history = []
    try:
        for size in graph_sizes:
            print(f"Benchmarking Graph Size {size}", end="")
            for _ in range(instance_num):
                print(".")
                # create graph
                #graph = random_lfr_graph(
                #    size,
                #    #community_size=50,
                #    mu=0.3,
                #    average_degree=5,
                #    rng=rng,
                #)
                graph = random_fcs_graph(
                    size,
                    community_size=50,
                    mu=0.3,
                    average_degree=5,
                    rng=rng,
                )
                # benchmark edgeqlouvain
                solver = qlouvain.QLouvainSG(graph, rng=rng)
                stats = solver.run_with_tracking()

                # append stats
                record = asdict(stats)
                record["n"] = size
                history.append(record)
            print()

    except KeyboardInterrupt:
        print("Benchmarking Interrupted")

    # write stats to disk
    df = pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )

    if os.path.exists(dest):
        existing_df = pd.read_json(dest)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_json(dest, orient="records")


# plotting
plot_dest = Path("edgeq.png")
quantum_factor = 2


def plot():
    # Read dataframe from disk
    df = pd.read_json(dest)

    # compute combined query costs of quantum search
    c = df["quantum_expected_classical_queries"]
    q = df["quantum_expected_quantum_queries"]
    df["quantum_cost"] = c + quantum_factor * q

    # Group by 'n' and calculate mean and standard deviation
    df_mean = df.groupby("n").mean()
    df_std = df.groupby("n").std()
    # df_grouped = df.groupby("n").agg([("mean"), "std"])

    # debug
    # df_grouped = df
    # df_grouped = df.groupby("n").agg("mean")
    # print(df_mean)
    print(df_mean.to_string())

    # Create line plot using Plotly
    # fig = px.line(
    #     df_mean,
    #     x=df_mean.index,
    #     y=df_mean["quantum_cost"],
    #     error_y=df_std["quantum_cost"],
    #     # df_grouped,
    #     # x=df_grouped.index,
    #     # y=df_grouped[("quantum_cost","mean")],
    #     # y=df_grouped[("quantum_cost", "mean")],
    #     # error_y=df_grouped[("quantum_cost", "std")],
    #     title="Edgeq Louvain Results",
    # )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_mean.index,
            y=df_mean["quantum_cost"],
            error_y=dict(type="data", array=df_std["quantum_cost"]),
            mode="lines",
            name="quantum_cost",
        )
    )

    # Add another line for 'classical_actual_queries'
    fig.add_trace(
        go.Scatter(
            x=df_mean.index,
            y=df_mean["classical_actual_queries"],
            # error_y=df_std["classical_actual_queries"],
            error_y=dict(type="data", array=df_std["classical_actual_queries"]),
            # x=df_grouped.index,
            # y=df_mean[("classical_actual_queries", "mean")],
            # error_y=dict(
            #     type="data", array=df_std[("classical_actual_queries", "std")]
            # ),
            mode="lines",
            name="classical_actual_queries",
        )
    )

    # Update x and y axes to log scale
    fig.update_xaxes(type="log", tickformat=".2e")
    fig.update_yaxes(type="log", tickformat=".2e")

    fig.show()


if __name__ == "__main__":
    bench()
    plot()
