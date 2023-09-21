"""
This module provides the Cora dataset.

For background on the dataset see https://www.openicpsr.org/openicpsr/project/100859/version/V1/view.
"""

import os

import networkx as nx
import pandas as pd


def get_graph() -> nx.Graph:
    """Return a NetworkX graph that represents the Cora dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "cora_data")
    nodes = pd.read_csv(os.path.join(data_path, "cora.content"), sep="\t", header=None, usecols=[0, 1434], names=["node", "subject"])
    edges = pd.read_csv(os.path.join(data_path, "cora.cites"), sep="\t", header=None, names=["target", "source"])

    G = nx.from_pandas_edgelist(edges)
    subject_map = {row["node"]: row["subject"] for i, row in nodes.iterrows()}
    nx.set_node_attributes(G, subject_map, "subject")

    return G
