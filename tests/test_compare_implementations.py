import random

import networkx as nx

from community_detection.louvain import louvain
from community_detection.quality_functions import Modularity, QualityFunction
from community_detection.utils import Partition

PRECISION = 1e-15

# Don't let black destroy the manual formatting in this document:
# fmt: off

def test_louvain_modularity_comparison_networkx_karate_club() -> None:
    """Compare our implementation of Modularity and the Louvain algorithm with the ones in NetworkX."""
    # This test uses the so-called Zachary’s Karate Club graph, which represents the social interactions in a Karate
    # sports club. The nodes represent two trainers and 32 students, the edges represent interactions between two
    # people.
    G = nx.karate_club_graph()

    # As the louvain algorithm is randomized, seed the RNG with a static value so that tests are reproducible.
    # Those seeds for our, respectively NetworkX' implementation produce the same result
    OUR_SEED = 0
    NX_SEED = 0

    random.seed(OUR_SEED)

    # We use modularity as quality function, with a resolution of 1.
    𝓗: QualityFunction[int] = Modularity(1)
    𝓟 = louvain(G, 𝓗, weight="weight")
    print(f"Our partition:       {𝓟.as_set()=}")

    # We compare the result to a partition calculated by the NetworkX library.
    # Due to the randomized nature of the louvain algorithm, we need to supply the implementation with a seed
    # so that the result stays consistent. Otherwise, the calculated communities will change between runs!
    𝓠 = Partition.from_partition(G, nx.community.louvain_communities(G, weight="weight", resolution=1, seed=NX_SEED), weight="weight")
    print(f"Reference partition: {𝓠.as_set()=}")

    # The following function uses NetworkX' implementation of modularity and makes it available so that we can use it
    # as a reference implementation to compare the values calculated by our implementation against.
    def nx_mod(𝓡: Partition[int]) -> float:
        mod: float = nx.community.modularity(G, 𝓡.as_set(), weight="weight", resolution=1)
        return mod

    # Save modularities calculated by our and NX' modularity functions of partitions calculated by us and NetworkX.
    olom, olnm, nlom, nlnm = 𝓗(𝓟), nx_mod(𝓟), 𝓗(𝓠), nx_mod(𝓠)

    print("Final modularities   | our Louvain impl. | NX' Louvain impl.")
    print(f"Our modularity impl. |      {olom:03.10f} |      {nlom:03.10f} ")
    print(f"NX' modularity impl. |      {olnm:03.10f} |      {nlnm:03.10f} ")

    # Both rows match: Modularity values calculated by us and NetworkX match
    assert abs(olom - olnm) < PRECISION, "Our and NX' modularity implementations don't match for the result of our Louvain implementation!"
    assert abs(nlom - nlnm) < PRECISION, "Our and NX' modularity implementations don't match for the result of NX' Louvain implementation!"
    # Both columns match: Partitions calculated by us and NetworkX have equal modularities
    assert abs(olom - nlom) < PRECISION, "The modularity (our implementation) does not match between our and NetworkX' implementation of the Louvain algorithm!"
    assert abs(olnm - nlnm) < PRECISION, "The modularity (NX' implementation) does not match between our and NetworkX' implementation of the Louvain algorithm!"

    # With the seed of 0 chosen above, the communities calculated by us and NetworkX even match:
    assert 𝓟.as_set() == 𝓠.as_set(), "The communities calculated by our implementation does not match the ones calculated by NetworkX's implementation of the Louvain algorithm!"


def test_louvain_modularity_comparison_networkx_jazz_musicians() -> None:
    """Compare our implementation of Modularity and the Louvain algorithm with the ones in NetworkX."""
    # This test uses the "Jazz Musician Dataset", which represents collaborations between jazz musicians.
    # The nodes represent musicians, the edges represent collaborations between two people.
    import datasets.jazz
    G = datasets.jazz.get_graph()

    # As the louvain algorithm is randomized, seed the RNG with a static value so that tests are reproducible.
    # Those seeds for our, respectively NetworkX' implementation produce the same result
    OUR_SEED = 43
    NX_SEED = 4

    random.seed(OUR_SEED)

    # We use modularity as quality function, with a resolution of 1.
    𝓗: QualityFunction[int] = Modularity(1)
    𝓟 = louvain(G, 𝓗, weight="weight")
    print(f"Our partition:       {𝓟.as_set()=}")

    # We compare the result to a partition calculated by the NetworkX library.
    # Due to the randomized nature of the louvain algorithm, we need to supply the implementation with a seed
    # so that the result stays consistent. Otherwise, the calculated communities will change between runs!
    𝓠 = Partition.from_partition(G, nx.community.louvain_communities(G, weight="weight", resolution=1, seed=NX_SEED), weight="weight")
    print(f"Reference partition: {𝓠.as_set()=}")

    # The following function uses NetworkX' implementation of modularity and makes it available so that we can use it
    # as a reference implementation to compare the values calculated by our implementation against.
    def nx_mod(𝓡: Partition[int]) -> float:
        mod: float = nx.community.modularity(G, 𝓡.as_set(), weight="weight", resolution=1)
        return mod

    # Save modularities calculated by our and NX' modularity functions of partitions calculated by us and NetworkX.
    olom, olnm, nlom, nlnm = 𝓗(𝓟), nx_mod(𝓟), 𝓗(𝓠), nx_mod(𝓠)

    print("Final modularities   | our Louvain impl. | NX' Louvain impl.")
    print(f"Our modularity impl. |      {olom:03.10f} |      {nlom:03.10f} ")
    print(f"NX' modularity impl. |      {olnm:03.10f} |      {nlnm:03.10f} ")

    # Both rows match: Modularity values calculated by us and NetworkX match
    assert abs(olom - olnm) < PRECISION, "Our and NX' modularity implementations don't match for the result of our Louvain implementation!"
    assert abs(nlom - nlnm) < PRECISION, "Our and NX' modularity implementations don't match for the result of NX' Louvain implementation!"
    # Both columns match: Partitions calculated by us and NetworkX have equal modularities
    assert abs(olom - nlom) < PRECISION, "The modularity (our implementation) does not match between our and NetworkX' implementation of the Louvain algorithm!"
    assert abs(olnm - nlnm) < PRECISION, "The modularity (NX' implementation) does not match between our and NetworkX' implementation of the Louvain algorithm!"

    # With the seeds chosen above, the communities calculated by us and NetworkX even match:
    assert 𝓟.as_set() == 𝓠.as_set(), "The communities calculated by our implementation does not match the ones calculated by NetworkX's implementation of the Louvain algorithm!"
