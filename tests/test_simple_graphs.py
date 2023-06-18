import networkx as nx

from ..leiden import leiden
from ..louvain import louvain
from ..utils import *


def test_louvain_barbell():
    """
    This test uses a (5, 2) barbell graph, that is two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    # We use the modularity as quality function, with a resolution of 2.
    𝓗 = Modularity(2)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.sets == [{7, 8, 9, 10, 11}, {0, 1, 2, 3, 4}, {5, 6}]


def test_leiden_barbell():
    """
    This test uses a (5, 2) barbell graph, that is two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    # We use the modularity as quality function, with a resolution of 2.
    𝓗 = Modularity(2)
    𝓠 = leiden(G, 𝓗)

    assert 𝓠.sets == [{7, 8, 9, 10, 11}, {0, 1, 2, 3, 4}, {5, 6}]
