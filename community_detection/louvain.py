"""
Implementation of the Louvain algorithm community detection.

This implementation follows the outline provided in the supplementary material of the paper "From Louvain to Leiden:
guaranteeing well-connected communities" by V.A. Traag, L. Waltman and N.J. van Eck.
"""

from typing import TypeVar

from networkx import Graph

from .quality_metrics import QualityMetric
from .utils import Partition, aggregate_graph, argmax, flatₚ

T = TypeVar("T")


def louvain(G: Graph, 𝓗: QualityMetric[T], 𝓟: Partition[T] | None = None) -> Partition[T]:
    """Perform the Louvain algorithm for community detection."""
    # If there is no partition given, start with every node in its' own community
    if 𝓟 is None:
        𝓟 = Partition.singleton_partition(G)

    # Remember the original graph
    G_orig = G

    while True:
        # First phase: Move nodes locally
        𝓟 = move_nodes(G, 𝓟, 𝓗)

        # When every community consists of a single node, terminate,
        # returning the flattened partition, as given by 𝓟.
        if len(𝓟) == len(G.nodes):
            return Partition.from_partition(G_orig, flatₚ(𝓟))

        # Second phase: Aggregation of the network
        # Create the aggregate graph of G based on the partition 𝓟
        G = aggregate_graph(G, 𝓟)
        # And update 𝓟 to be a singleton partition of G, i.e. every node in the aggregate graph G
        # is assigned to its own community.
        𝓟 = Partition.singleton_partition(G)


def move_nodes(G: Graph, 𝓟: Partition[T], 𝓗: QualityMetric[T]) -> Partition[T]:
    """Perform node moves to communities as long as the quality metric can be improved by moving."""
    # This is the python form of a "do-while" loop
    while True:
        # Store current ("o" for "old") quality function value
        𝓗ₒ = 𝓗(G, 𝓟)

        for v in G.nodes:
            # Find best community for node `v` to be in, potentially creating a new community.
            # Cₘ is the optimal community, 𝛥𝓗 is the increase of 𝓗 over 𝓗ₒ, reached at Cₘ.
            (Cₘ, 𝛥𝓗, _) = argmax(lambda C: 𝓗.delta(G, 𝓟, v, C, 𝓗ₒ), [*𝓟, set()])

            # If we get a strictly better value, assign v to community Cₘ
            if 𝛥𝓗 > 0:
                𝓟.move_node(v, Cₘ)

        # If no further improvement can be made, we're done and return the current partition
        if 𝓗(G, 𝓟) <= 𝓗ₒ:
            return 𝓟
