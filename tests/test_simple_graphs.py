"""Test the Louvain and Leiden algorithms on simple graphs."""

import networkx as nx

from community_detection.leiden import leiden
from community_detection.louvain import louvain
from community_detection.quality_metrics import CPM, Modularity
from community_detection.utils import freeze

# Below are a few tests written for simple graphs (currently only ones for the (5,2) barbell graph),
# where one possible / logical 

BARBELL_PARTS = freeze([{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}])

def test_louvain_barbell_modularity():
    """
    Test the Louvain algorithm with modularity as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    𝓗 = Modularity(1)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_PARTS


def test_leiden_barbell_modularity():
    """
    Test the Leiden algorithm with modularity as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    𝓗 = Modularity(1.5)
    𝓠 = leiden(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_PARTS


def test_louvain_barbell_cpm():
    """
    Test the Louvain algorithm with CPM as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    # The following resolution parameter for the CPM was found using binary serach on the interval [0.95, 1.05].
    𝓗 = CPM(0.9999999999999986)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_PARTS


def test_leiden_barbell_cpm():
    """
    Test the Leiden algorithm with CPM as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    𝓗 = CPM(1)
    𝓠 = leiden(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_PARTS
