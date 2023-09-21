"""
This module provides the jazz musician collaboration dataset.

For background on the dataset see https://www.worldscientific.com/doi/abs/10.1142/S0219525903001067.
"""

import os

import networkx as nx


def get_graph() -> nx.Graph:
    """Return a NetworkX graph that represents the Jazz musician dataset."""
    path = os.path.join(os.path.dirname(__file__), "jazz_data/out.arenas-jazz")
    G = nx.read_edgelist(path, comments="%", create_using=nx.Graph(), nodetype=int)

    return G
