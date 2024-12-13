import pandas as pd
import numpy as np
import math
import plotly.express as px

def plot_histogram_log(ligands, predictions_log):
    """
    Plots a histogram of predicted affinities in the log domain.

    Args:
        ligands (list or np.ndarray): List of ligand SMILES.
        predictions_log (list or np.ndarray): Predicted affinities in the log domain.

    Returns:
        plotly.graph_objects.Figure: Interactive bar plot.
    """
    # Ensure inputs are one-dimensional
    ligands = np.array(ligands).flatten()
    predictions_log = np.array(predictions_log).flatten()

    # Combine predictions and ligands into a DataFrame
    predictions_df = pd.DataFrame({
        "Ligand SMILES": ligands,
        "Predicted Affinity (log)": predictions_log
    })

    # Define bin edges for the histogram
    bin_edges = np.linspace(predictions_log.min(), predictions_log.max(), 11)
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]

    # Assign each prediction to a bin
    predictions_df["Bin"] = pd.cut(
        predictions_df["Predicted Affinity (log)"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        ordered=True
    )

    # Format hover data
    def format_hover_data(x):
        formatted = [
            f"{smiles}: {affinity:.2f}" for smiles, affinity in zip(x["Ligand SMILES"], x["Predicted Affinity (log)"])
        ]
        return "<br>".join(formatted[:10]) + ("<br>and more..." if len(formatted) > 10 else "")

    hist_data = predictions_df.groupby("Bin", observed=False).apply(
        lambda group: {
            "Count": len(group),
            "Details": format_hover_data(group)
        }
    ).reset_index(name="AggregatedData")

    hist_data["Count"] = hist_data["AggregatedData"].apply(lambda x: x["Count"])
    hist_data["Details"] = hist_data["AggregatedData"].apply(lambda x: x["Details"])
    hist_data = hist_data.drop(columns=["AggregatedData"])

    # Create the plot
    fig = px.bar(
        hist_data,
        x="Bin",
        y="Count",
        hover_data={"Details": True},
        title="Histogram of Predicted Affinities (IC50 in log domain)",
        labels={"Bin": "Affinity Range (log)", "Count": "Frequency"},
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title="Predicted Affinity Range (log)",
        yaxis_title="Frequency",
        hoverlabel=dict(font_size=12, align="left"),
        bargap=0.1,
        title=dict(x=0.5, xanchor="center"),
        font=dict(family="Arial", size=12)
    )

    fig.show()


def plot_histogram_nonlog(ligands, predictions):
    """
    Plots a histogram of predicted affinities in the non-log domain.

    Args:
        ligands (list): List of ligand SMILES.
        predictions (list): Predicted affinities in the non-log domain.

    Returns:
        plotly.graph_objects.Figure: Interactive bar plot.
    """
    # Ensure inputs are one-dimensional
    ligands = np.array(ligands).flatten()
    predictions = np.array(predictions).flatten()

    # Combine predictions and ligands into a DataFrame
    predictions_df = pd.DataFrame({
        "Ligand SMILES": ligands,
        "Predicted Affinity (nM)": predictions
    })

    # Custom bin edges
    bin_edges = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, math.ceil(max(predictions))]
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]

    # Assign each prediction to a bin
    predictions_df["Bin"] = pd.cut(
        predictions_df["Predicted Affinity (nM)"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        ordered=True
    )

    # Format hover data
    def format_hover_data(x):
        formatted = [
            f"{smiles}: {affinity:.2f} nM" for smiles, affinity in zip(x["Ligand SMILES"], x["Predicted Affinity (nM)"])
        ]
        return "<br>".join(formatted[:10]) + ("<br>and more..." if len(formatted) > 10 else "")

    hist_data = predictions_df.groupby("Bin", observed=False).apply(
        lambda group: {
            "Count": len(group),
            "Details": format_hover_data(group)
        }
    ).reset_index(name="AggregatedData")

    hist_data["Count"] = hist_data["AggregatedData"].apply(lambda x: x["Count"])
    hist_data["Details"] = hist_data["AggregatedData"].apply(lambda x: x["Details"])
    hist_data = hist_data.drop(columns=["AggregatedData"])

    # Create the plot
    fig = px.bar(
        hist_data,
        x="Bin",
        y="Count",
        hover_data={"Details": True},
        title="Histogram of Predicted Affinities (IC50 in nM)",
        labels={"Bin": "Affinity Range (nM)", "Count": "Frequency"},
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title="Predicted Affinity Range (nM)",
        yaxis_title="Frequency",
        hoverlabel=dict(font_size=12, align="left"),
        bargap=0.1,
        title=dict(x=0.5, xanchor="center"),
        font=dict(family="Arial", size=12)
    )

    fig.show()

