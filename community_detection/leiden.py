"""
Implementation of the Leiden algorithm for community detection.

This implementation follows the outline provided in the supplementary material of the paper "From Louvain to Leiden:
guaranteeing well-connected communities" by V.A. Traag, L. Waltman and N.J. van Eck.
"""

from collections.abc import Set
from math import exp
from random import choices, shuffle
from typing import TypeVar

import networkx as nx
from networkx import Graph

from .quality_functions import QualityFunction
from .utils import DataKeys as Keys
from .utils import Partition, argmax, freeze, node_total, preprocess_graph

T = TypeVar("T")


def leiden(
    G: Graph, H: QualityFunction[T], P: Partition[T] | None = None, theta: float = 0.3, gamma: float = 0.05, weight: str | None = None
) -> Partition[T]:
    """
    Perform the Leiden algorithm for community detection.

    Parameters
    ----------
    G : Graph
        The graph / network to process.
    H : QualityFunction[T]
        A quality function to optimize.
    P : Partition[T], optional
        A partition to use as basis, leave at the default of `None` when none is available.
    theta : float, optional
        The theta parameter of the Leiden method, which determines the randomness in the refinement phase of the Leiden
        algorithm, default value of 0.3.
    gamma : float, optional
        The gamma parameter of the Leiden method, default value of 0.05.

    :returns: A partition of G into communities.
    """
    # For every edge, assign an edge weight attribute of 1, if no weight is set yet.
    G = preprocess_graph(G, weight)

    # If there is a partition given, use it, else start with a singleton partition of the graph
    if P:
        P = Partition.from_partition(G, P, Keys.WEIGHT)
    else:
        P = Partition.singleton_partition(G, Keys.WEIGHT)

    # Remember the Previous partition, in order to terminate when the sequence of partitions becomes stationary.
    # This isn't handled by the provided pseudocode, but this can happen, if gamma is chosen too large for the given graph.
    # In this case, refine_partition will always return the singleton partition of G, which will lead to an endless loop, as G will  become
    # the aggregate graph of G with respect to the singleton partition, which is just G again.
    # Thus, P will also be set to the value it had before, and, as we got to refine_partition, len(P) != G.order() and thus, we'd get an
    # infinite loop.
    P_p = None

    while True:
        P = move_nodes_fast(G, P, H)

        # When every community consists of a single node only, terminate, returning the flat partition given by P.
        # Also terminate, if the sequence of partition generated becomes stationary.
        if len(P) == G.order() or P == P_p:
            # Return the partition P in terms of the original graph, which was passed to this function
            return P.flatten()

        # Remember partition for termination check.
        P_p = P

        # Refine the partition created by fast local moving, potentially splitting a community into multiple parts
        P_r = refine_partition(G, P, H, theta, gamma)
        # Create the aggregate graph of G based on P_r ...
        G = P_r.aggregate_graph()

        # ... but maintain partition P, that is, lift it to the aggregate graph.
        # The following lines are equivalent to, but way faster than
        # `partitions = [{v for v in G.nodes if G.nodes[v][Keys.NODES] <= C} for C in P]`.
        partitions: dict[int, set[T]] = {id: set() for id in range(len(P))}
        # Iterate through the aggregate graph's nodes
        for v_agg, nodes in G.nodes(data=Keys.NODES):
            # Get the id of the community that the nodes collected in this super node were part of
            community_id = P._node_part[next(iter(nodes))]
            # Note that down in the partitions dict
            partitions[community_id] = partitions[community_id] | {v_agg}
        # Now, discard the indices and produce the list of values, i.e. the lifted partition
        partitions_l: list[set[T]] = list(partitions.values())

        P = Partition.from_partition(G, partitions_l, Keys.WEIGHT)


def move_nodes_fast(G: Graph, P: Partition[T], H: QualityFunction[T]) -> Partition[T]:
    """
    Perform fast local node moves to communities to improve the partition's quality.

    For every node, greedily move it to a neighboring community, maximizing the improvement in the partition's quality.
    """
    # Create a queue to visit all nodes in random order.
    Q = list(G.nodes)
    shuffle(Q)

    while True:
        # Determine next node to visit by popping first node in the queue
        v = Q.pop(0)

        # Find an optimal community for node `v` to be in, potentially creating a new community.
        # C_m is the optimal community, delta_H is the increase of H over H_0, reached at C_m.
        (C_m, delta_H, _) = argmax(lambda C: H.delta(P, v, C), [*P.adjacent_communities(v), set()])

        # If we can achieve a strict improvement
        if delta_H > 0:
            # Move node v to community C_m
            P.move_node(v, C_m)

            # Identify neighbors of v that are not in C_m
            N = {u for u in G[v] if u not in C_m}

            # Visit these neighbors as well
            Q.extend(N - set(Q))

        # If queue is empty, return P
        if len(Q) == 0:
            return P


def refine_partition(G: Graph, P: Partition[T], H: QualityFunction[T], theta: float, gamma: float) -> Partition[T]:
    """Refine all communities by merging repeatedly, starting from a singleton partition."""
    # Assign each node to its own community
    P_r: Partition[T] = Partition.singleton_partition(G, Keys.WEIGHT)

    # Visit all communities
    for C in P:
        # refine community
        P_r = merge_nodes_subset(G, P_r, H, theta, gamma, C)

    return P_r


def merge_nodes_subset(G: Graph, P: Partition[T], H: QualityFunction[T], theta: float, gamma: float, S: Set[T]) -> Partition[T]:
    """Merge the nodes in the subset S into one or more sets to refine the partition P."""
    size_s = node_total(G, S)

    R = {
        v for v in S
          if nx.cut_size(G, [v], S - {v}, weight=Keys.WEIGHT) >= gamma * node_total(G, v) * (size_s - node_total(G, v))
    }  # fmt: skip

    for v in R:
        # If v is in a singleton community, i.e. is a node that has not yet been merged
        if len(P.node_community(v)) == 1:
            # Consider only well-connected communities
            T = freeze([
                C for C in P
                  if C <= S and nx.cut_size(G, C, S - C, weight=Keys.WEIGHT) >= gamma * (node_total(G, C) * (size_s - node_total(G, C)))
            ])  # fmt: skip

            # Now, choose a random community to put v into
            # We use python's random.choices for the weighted choice, as this is easiest.

            # Have a list of pairs of communities in T together with the improvement (delta_H) of moving v to the community
            # Only consider communities for which the quality function doesn't degrade, if v is moved there
            communities = [(C, delta_H) for (C, delta_H) in ((C, H.delta(P, v, C)) for C in T) if delta_H >= 0]
            # Calculate the weights for the random choice using the delta_H values
            weights = [exp(delta_H / theta) for (C, delta_H) in communities]

            # Finally, choose the new community
            # Use [0][0] to extract the community, since choices returns a list, containing a single (C, delta_H) tuple
            C_new = choices(communities, weights=weights, k=1)[0][0]

            # And move v there
            P.move_node(v, C_new)

    return P
