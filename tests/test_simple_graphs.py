import networkx as nx

from ..leiden import leiden
from ..louvain import louvain
from ..utils import *


# Below are a few tests written for simple graphs (currently only ones for the (5,2) barbell graph),
# where one possible / logical 

BARBELL_PARTS = freeze([{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}])

def test_louvain_barbell_modularity():
    """
    This test uses a (5, 2) barbell graph, that is two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    𝓗 = Modularity(1)
    𝓠 = louvain(G, 𝓗)

    print("If you can read this, this test probably failed. Recall that these are randomized algorithms, " + \
          "so the exact partitions produce may not match the expected ones every now and then. Consider rerunning this test.")
    assert 𝓠.as_set() == BARBELL_PARTS


def test_leiden_barbell_modularity():
    """
    This test uses a (5, 2) barbell graph, that is two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    𝓗 = Modularity(1.5)
    𝓠 = leiden(G, 𝓗)

    print("If you can read this, this test probably failed. Recall that these are randomized algorithms, " + \
          "so the exact partitions produce may not match the expected ones every now and then. Consider rerunning this test.")
    assert 𝓠.as_set() == BARBELL_PARTS


def test_louvain_barbell_cpm():
    """
    This test uses a (5, 2) barbell graph, that is two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    # The following resolution parameter for the CPM was found using binary serach on the interval [0.95, 1.05].
    𝓗 = CPM(0.9999999999999986)
    𝓠 = louvain(G, 𝓗)

    print("If you can read this, this test probably failed. Recall that these are randomized algorithms, " + \
          "so the exact partitions produce may not match the expected ones every now and then. Consider rerunning this test.")
    assert 𝓠.as_set() == BARBELL_PARTS


def test_leiden_barbell_cpm():
    """
    This test uses a (5, 2) barbell graph, that is two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.classic.barbell_graph(5, 2)

    𝓗 = CPM(1)
    𝓠 = leiden(G, 𝓗)

    print("If you can read this, this test probably failed. Recall that these are randomized algorithms, " + \
          "so the exact partitions produce may not match the expected ones every now and then. Consider rerunning this test.")
    assert 𝓠.as_set() == BARBELL_PARTS
