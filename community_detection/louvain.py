"""
Implementation of the Louvain algorithm community detection.

This implementation follows the outline provided in the supplementary material of the paper "From Louvain to Leiden:
guaranteeing well-connected communities" by V.A. Traag, L. Waltman and N.J. van Eck.
"""

from typing import TypeVar
from random import shuffle

from networkx import Graph

from .quality_metrics import QualityMetric
from .utils import Partition, argmax, preprocess_graph

T = TypeVar("T")


def louvain(G: Graph, 𝓗: QualityMetric[T], 𝓟: Partition[T] | None = None) -> Partition[T]:
    """Perform the Louvain algorithm for community detection."""
    # For every edge, assign an edge weight attribute of 1, if no weight is set yet.
    G = preprocess_graph(G, "weight")

    # If there is no partition given, start with every node in its' own community
    if 𝓟 is None:
        𝓟 = Partition.singleton_partition(G, "weight")

    while True:
        # First phase: Move nodes locally
        𝓟 = move_nodes(G, 𝓟, 𝓗)

        # When every community consists of a single node, terminate, returning the flattened partition, as given by 𝓟.
        if len(𝓟) == G.order():
            return 𝓟.flatten()

        # Second phase: Aggregation of the network
        # Create the aggregate graph of G based on the partition 𝓟
        G = 𝓟.aggregate_graph()
        # And update 𝓟 to be a singleton partition of G, i.e. every node in the aggregate graph G is assigned to its own community.
        𝓟 = Partition.singleton_partition(G, "weight")


def move_nodes(G: Graph, 𝓟: Partition[T], 𝓗: QualityMetric[T]) -> Partition[T]:
    """Perform node moves to communities as long as the quality metric can be improved by moving."""
    # This is the python form of a "do-while" loop
    while True:
        Q = list(G.nodes)
        shuffle(Q)
        improved = False
        for v in Q:
            # Find best community for node `v` to be in, potentially creating a new community.
            # Cₘ is the optimal community, 𝛥𝓗 is the increase of 𝓗 over 𝓗ₒ (value at beginning of outer loop), reached by moving v into Cₘ.
            neighbor_communities = {frozenset(𝓟._sets[i]) for i in {𝓟._node_part[u] for u in G.neighbors(v)}}
            (Cₘ, 𝛥𝓗, _) = argmax(lambda C: 𝓗.delta(G, 𝓟, v, C, "weight"), [*neighbor_communities, set()])

            # If we get a strictly better value, assign v to community Cₘ
            if 𝛥𝓗 > 0:
                improved = True
                𝓟.move_node(v, Cₘ)

        # If no further improvement can be made, we're done and return the current partition
        if not improved:
            return 𝓟
