"""
Implementation of the Louvain algorithm community detection.

This implementation follows the outline provided in the supplementary material of the paper "From Louvain to Leiden:
guaranteeing well-connected communities" by V.A. Traag, L. Waltman and N.J. van Eck.
"""

from random import shuffle
from typing import TypeVar

from networkx import Graph

from .quality_functions import QualityFunction
from .utils import DataKeys as Keys
from .utils import Partition, argmax, preprocess_graph

T = TypeVar("T")


def louvain(G: Graph, H: QualityFunction[T], P: Partition[T] | None = None, weight: None | str = None) -> Partition[T]:
    """
    Perform the Louvain algorithm for community detection.

    Parameters
    ----------
    G : Graph
        The graph / network to process.
    H : QualityFunction[T]
        A quality function to optimize.
    P : Partition[T], optional
        A partition to use as basis, leave at the default of `None` when none is available.

    :returns: A partition of G into communities.
    """
    # For every edge, assign an edge weight attribute of 1, if no weight is set yet.
    G = preprocess_graph(G, weight)

    # If there is a partition given, use it, else start with every node in its own community
    if P:
        P = Partition.from_partition(G, P, Keys.WEIGHT)
    else:
        P = Partition.singleton_partition(G, Keys.WEIGHT)

    while True:
        # First phase: Move nodes locally
        P = move_nodes(G, P, H)

        # When every community consists of a single node, terminate, returning the flattened partition, as given by P.
        if len(P) == G.order():
            return P.flatten()

        # Second phase: Aggregation of the network
        # Create the aggregate graph of G based on the partition P
        G = P.aggregate_graph()
        # And update P to be a singleton partition of G, i.e. every node in the aggregate graph G is assigned to its own community.
        P = Partition.singleton_partition(G, Keys.WEIGHT)


def move_nodes(G: Graph, P: Partition[T], H: QualityFunction[T]) -> Partition[T]:
    """Perform node moves to communities as long as the quality function can be improved by moving."""
    # This is the python form of a "do-while" loop
    while True:
        Q = list(G.nodes)
        shuffle(Q)
        improved = False
        for v in Q:
            # Find an optimal community for node `v` to be in, potentially creating a new community.
            # C_m is the optimal community, delta_H is the increase of H over H_0 (value at beginning of outer loop),
            # reached by moving v into C_m.
            (C_m, delta_H, _) = argmax(lambda C: H.delta(P, v, C), [*P.adjacent_communities(v), set()])

            # If we get a strictly better value, assign v to community C_m
            if delta_H > 0:
                improved = True
                P.move_node(v, C_m)

        # If no further improvement can be made, we're done and return the current partition
        if not improved:
            return P
