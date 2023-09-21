"""
This module provides the Arxiv GR-QC collaboration network.

For background on the dataset see http://snap.stanford.edu/data/ca-GrQc.html.
"""

import os

import networkx as nx
import pandas as pd


def get_graph() -> nx.Graph:
    """Return a NetworkX graph that represents the Arxiv GR-QC dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "relativity_data")
    edges = pd.read_csv(os.path.join(data_path, "ca-GrQc.txt"), sep="\t", header=None, names=["target", "source"], comment="#")

    G = nx.from_pandas_edgelist(edges)

    return G
