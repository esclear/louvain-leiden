from math import isnan

import networkx as nx

from ..leiden import leiden
from ..louvain import louvain
from ..quality_metrics import CPM, Modularity
from ..utils import *

PRECISION = 1e-15


def test_modularity_trivial_values():
    """
    Test modularity calculation for special graphs and partitions to see if the values match our expectation.
    """
    C = nx.complete_graph(10)
    𝓟 = Partition(C, [{i for i in range(10)}])
    E = nx.empty_graph(10)
    𝓠 = Partition(E, [{i} for i in range(10)])

    𝓗 = Modularity(1)

    assert 0.0 == 𝓗(C, 𝓟)
    assert abs(-0.1 - 𝓗(C, 𝓠)) < PRECISION

    # For empty graphs, the modularity is not defined. We return NaN in this case:
    assert isnan(𝓗(E, 𝓟))
    assert isnan(𝓗(E, 𝓠))


def test_cpm_trivial_values():
    """
    Test modularity calculation for special graphs and partitions to see if the values match the expectation.
    """
    C = nx.complete_graph(10)
    𝓟 = Partition(C, [{i for i in range(10)}])
    E = nx.empty_graph(10)
    𝓠 = Partition(E, [{i} for i in range(10)])

    𝓗 = CPM(0.25)

    # Values calculated manually for γ = 0.25:
    assert -11.25 == 𝓗(E, 𝓟)  # The empty graph (no edges) with the trivial partition has CPM -11.25
    assert   0.00 == 𝓗(E, 𝓠)  # Empty graph with singleton partition has CPM 0 (better than the trivial partition)
    assert   0.00 == 𝓗(C, 𝓠)  # Complete graph K_10 with singleton partition has CPM 0
    assert  33.75 == 𝓗(C, 𝓟)  # The graph K_10 with the trivial partition has CPM 33.75 (improves singleton partition)


def test_modularity_comparison_networkx():
    """
    This test compares our implementation of Modularity and the Louvain algorithm with the ones in NetworkX.
    """
    # This test uses the so-called Zachary’s Karate Club graph, which represents the social interactions in a Karate
    # sports club. The nodes represent two trainers and 32 students, the edges represent interactions between two
    # people.
    G = nx.karate_club_graph()

    # We use modularity as quality function, with a resolution of 1.
    𝓗 = Modularity(1)
    𝓟 = louvain(G, 𝓗)

    # We compare the result to a partition calculated by the NetworkX library.
    # Due to the randomized nature of the louvain algorithm, we need to supply the implementation with a seed
    # so that the result stays consistent. Otherwise the calculated communities will change between runs!
    𝓠 = Partition(G, nx.community.louvain_communities(G, weight=None, resolution=1, seed=1))
    # The following lambda uses NetworkX' implementation of modularity and makes it available so that we can use it
    # to compare the values calculated by our implementation to the values calculated by NetworkX' implementation.
    nxMod = lambda 𝓡: nx.community.modularity(G, 𝓡.as_set(), weight=None, resolution=1)

    # Save modularities calculated by our and NX' modularity functions of partitions calculated by us and NetworkX.
    olom, olnm, nlom, nlnm = 𝓗(G, 𝓟), nxMod(𝓟), 𝓗(G, 𝓠), nxMod(𝓠)

    print(f"Final modularities   | our Louvain impl. | NX' Louvain impl.")
    print(f"Our modularity impl. |      {olom:03.10f} |      {nlom:03.10f} ")
    print(f"NX' modularity impl. |      {olnm:03.10f} |      {nlnm:03.10f} ")

    # Both rows match: Modularity values calculated by us and NetworkX match
    assert abs(olom - olnm) < PRECISION, "Our and NX' modularity implementations don't match for the result of our Louvain implementation!"
    assert abs(nlom - nlnm) < PRECISION, "Our and NX' modularity implementations don't match for the result of NX' Louvain implementation!"
    # Both columns match: Partitions calculated by us and NetworkX have equal modularities
    assert abs(olom - nlom) < PRECISION, "The modularity (our implementation) does not match between our and NetworkX' implementation of the Louvain algorithm!"
    assert abs(olnm - nlnm) < PRECISION, "The modularity (NX' implementation) does not match between our and NetworkX' implementation of the Louvain algorithm!"

    # With the seed of 1 chosen above, the communities calculated by us and NetworkX match:
    assert P.as_set() == Q.as_set(), "The communities calculated by our implementation does not match the ones calculated by NetworkX's implementation of the Louvain algorithm!"
