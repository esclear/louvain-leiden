"""Test the Louvain and Leiden algorithms on simple graphs."""

import networkx as nx
import random

from community_detection.leiden import leiden
from community_detection.louvain import louvain
from community_detection.quality_metrics import CPM, Modularity, QualityMetric
from community_detection.utils import freeze

# Below are a few tests written for simple graphs (currently only ones for the (5,2) barbell graph),  which are small enough for the
# results to be traceable by a human.
# The choice of this graph is inspired by the problem with greedy algorithms described on p. 6 of the supplementary material
# of the "From Louvain to Leiden" paper.
BARBELL_GOOD = freeze([{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}])
BARBELL_BAD = freeze([{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}])


def test_louvain_barbell_modularity() -> None:
    """
    Test the Louvain algorithm with modularity as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    # Due to the randomized nature of the algorithms, we seed the random number generator used so that we get the expected
    # communities. Due to the greedy nature of the louvain algorithm, we will never reach BARBELL_GOOD, but nodes 5 and 6
    # will often belong to one of the communities belonging to one or both of the complete graphs
    random.seed(10692)

    𝓗: QualityMetric[int] = Modularity(1)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_BAD


def test_leiden_barbell_modularity() -> None:
    """
    Test the Leiden algorithm with modularity as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    𝓗: QualityMetric[int] = Modularity(1.5)
    𝓠 = leiden(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_GOOD


def test_louvain_barbell_cpm() -> None:
    """
    Test the Louvain algorithm with CPM as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    # The following resolution parameter for the CPM was found using binary serach on the interval [0.95, 1.05].
    𝓗: QualityMetric[int] = CPM(0.9999999999999986)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_BAD


def test_leiden_barbell_cpm() -> None:
    """
    Test the Leiden algorithm with CPM as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    𝓗: QualityMetric[int] = CPM(0.9999999999999986)
    𝓠 = leiden(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_GOOD
