"""
Test the Louvain and Leiden algorithms on simple graphs.

The algorithms are tested against two small graphs, so that the results are traceable by a human.

A note regarding randomness:
Both the Louvain and the Leiden algorithm are randomized algorithms.
To have reproducible test results, every test in this file seeds the random number generator to a constant, predefined value.
This way, the tests are reproducible and the resulting partitions do not change in every test execution.
As we only have a handful of (relevant) reference partitions, some seeds are chosen so that the execution results in a certain partition,
as is the case with the example of the weighted (4,0) barbell graph in the later section of this file.
"""

import networkx as nx

from community_detection.leiden import leiden
from community_detection.louvain import louvain
from community_detection.quality_functions import CPM, Modularity, QualityFunction
from community_detection.utils import freeze

from .utils import seed_rng

#######################
# (5,2) BARBELL GRAPH #
#######################

# The choice of this unweighted (5,2) barbell graph is inspired by the problem with greedy algorithms described on p. 6 of the
# supplementary material of the "From Louvain to Leiden" paper.

BARBELL_COMS_AND_MID = freeze([{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}])
BARBELL_COMS_NO_MID = freeze([{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}])

# Contrary to the weighted example graph used below, it is *not possible* for the leiden algorithm to reach the `BARBELL_COMS_NO_MID`
# partition of the unweighted input graph.

# Due to the randomized nature of the algorithms, we seed the random number generator used so that we get the expected
# communities. Due to the greedy nature of the louvain algorithm, we will never reach BARBELL_GOOD, but nodes 5 and 6
# will often belong to one of the communities belonging to one or both of the complete graphs.


@seed_rng(0)
def test_louvain_barbell_modularity() -> None:
    """
    Test the Louvain algorithm with modularity as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    𝓗: QualityFunction[int] = Modularity(1)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_COMS_AND_MID


@seed_rng(0)
def test_leiden_barbell_modularity() -> None:
    """
    Test the Leiden algorithm with modularity as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    𝓗: QualityFunction[int] = Modularity(1.1)
    𝓠 = leiden(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_COMS_AND_MID


@seed_rng(0)
def test_louvain_barbell_cpm() -> None:
    """
    Test the Louvain algorithm with CPM as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    # The following resolution parameter for the CPM was found using binary search on the interval [0.95, 1.05].
    𝓗: QualityFunction[int] = CPM(0.9999999999999986)
    𝓠 = louvain(G, 𝓗)

    assert 𝓠.as_set() == BARBELL_COMS_AND_MID


@seed_rng(0)
def test_leiden_barbell_cpm() -> None:
    """
    Test the Leiden algorithm with CPM as the quality function on a (5,2) barbell graph.

    This graph consists of two complete graphs K_5, connected by a path of length 2.
    """
    G = nx.generators.barbell_graph(5, 2)

    𝓗: QualityFunction[int] = CPM(0.8)
    𝓠 = leiden(G, 𝓗, γ=0.8, θ=0.25)

    assert 𝓠.as_set() == BARBELL_COMS_AND_MID


################################
# WEIGHTED (4,0) BARBELL GRAPH #
################################

# The weighted (4,0) barbell graph used below is taken from page 6 of the supplementary material to the paper "From Louvain to Leiden".
# It admits a partition (called WEIGHTED_BARBELL_GOOD), which cannot be found by the Louvain algorithm, due to its greedy nature.
# The `test_leiden_*` tests below show that the Leiden algorithm can indeed reach that partition.


def _get_weighted_barbell_graph() -> nx.Graph:
    """Return the weighted (4,0) barbell graph found on p. 6 of the supplementary material of the "From Louvain to Leiden" paper."""
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 3),
        (0, 2, 1.5), (0, 3, 1.5), (0, 4, 1.5), (2, 3, 3), (2, 4, 3), (3, 4, 3),
        (1, 5, 1.5), (1, 6, 1.5), (1, 7, 1.5), (5, 6, 3), (5, 7, 3), (6, 7, 3)
    ])  # fmt: skip
    return G


# This graph can be partitioned into the following partitions (amongst others):
WEIGHTED_BARBELL_GOOD = freeze([{0, 2, 3, 4}, {1, 5, 6, 7}])
WEIGHTED_BARBELL_BAD = freeze([{2, 3, 4}, {0, 1}, {5, 6, 7}])


@seed_rng(0)
def test_louvain_weighted_barbell_modularity() -> None:
    """
    Test the Louvain algorithm with modularity as the quality function on a weighted (4,0) barbell graph.

    This graph consists of two complete graphs K_4, connected by a single edge.
    """
    G = _get_weighted_barbell_graph()

    𝓗: QualityFunction[int] = Modularity(1)
    𝓠 = louvain(G, 𝓗, weight="weight")

    assert 𝓠.as_set() == WEIGHTED_BARBELL_BAD

    # Also, rerun the algorithm to see whether the result changed (it shouldn't)
    𝓠 = louvain(G, 𝓗, 𝓠, weight="weight")

    assert 𝓠.as_set() == WEIGHTED_BARBELL_BAD


# This test proves that the Leiden algorithm *can arrive* at the WEIGHTED_BARBELL_GOOD partition, which cannot be reached by the
# greedy Louvain algorithm (cf. the Louvain and Leiden paper).
# The seed below leads to *this exact* partition (and not a partition of similar quality)
@seed_rng(0)
def test_leiden_weighted_barbell_modularity() -> None:
    """
    Test the Leiden algorithm with modularity as the quality function on a weighted (4,0) barbell graph.

    This graph consists of two complete graphs K_4, connected by a single edge.
    """
    G = _get_weighted_barbell_graph()

    𝓗: QualityFunction[int] = Modularity(0.75)
    𝓠 = leiden(G, 𝓗, weight="weight")

    assert 𝓠.as_set() == WEIGHTED_BARBELL_GOOD

    # Also, rerun the algorithm to see whether the result changed (it shouldn't)
    𝓠 = leiden(G, 𝓗, 𝓠, weight="weight")

    assert 𝓠.as_set() == WEIGHTED_BARBELL_GOOD


@seed_rng(0)
def test_louvain_weighted_barbell_cpm() -> None:
    """
    Test the Louvain algorithm with CPM as the quality function on a weighted (4,0) barbell graph.

    This graph consists of two complete graphs K_4, connected by a single edge.
    """
    G = _get_weighted_barbell_graph()

    # The following resolution parameter for the CPM was found using binary search on the interval [0.95, 1.05].
    𝓗: QualityFunction[int] = CPM(0.9999999999999986)
    𝓠 = louvain(G, 𝓗, weight="weight")

    assert 𝓠.as_set() == WEIGHTED_BARBELL_BAD


# Another seed, leading to the partition that the Louvain algorithm cannot reach.
@seed_rng(0)
def test_leiden_weighted_barbell_cpm() -> None:
    """
    Test the Leiden algorithm with CPM as the quality function on a weighted (4,0) barbell graph.

    This graph consists of two complete graphs K_4, connected by a single edge.
    """
    G = _get_weighted_barbell_graph()

    𝓗: QualityFunction[int] = CPM(0.5)
    𝓠 = leiden(G, 𝓗, weight="weight")

    assert 𝓠.as_set() == WEIGHTED_BARBELL_GOOD
