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
    G: Graph, 𝓗: QualityFunction[T], 𝓟: Partition[T] | None = None, θ: float = 0.3, γ: float = 0.05, weight: str | None = None
) -> Partition[T]:
    """
    Perform the Leiden algorithm for community detection.

    Parameters
    ----------
    G : Graph
        The graph / network to process.
    𝓗 : QualityFunction[T]
        A quality function to optimize.
    𝓟 : Partition[T], optional
        A partition to use as basis, leave at the default of `None` when none is available.
    θ : float, optional
        The θ parameter of the Leiden method, which determines the randomness in the refinement phase of the Leiden
        algorithm, default value of 0.3.
    γ : float, optional
        The γ parameter of the Leiden method, default value of 0.05.

    :returns: A partition of G into communities.
    """
    # For every edge, assign an edge weight attribute of 1, if no weight is set yet.
    G = preprocess_graph(G, weight)

    # If there is a partition given, use it, else start with a singleton partition of the graph
    if 𝓟:
        𝓟 = Partition.from_partition(G, 𝓟, Keys.WEIGHT)
    else:
        𝓟 = Partition.singleton_partition(G, Keys.WEIGHT)

    # Remember the Previous partition, in order to terminate when the sequence of partitions becomes stationary.
    # This isn't handled by the provided pseudocode, but this can happen, if γ is chosen too large for the given graph.
    # In this case, refine_partition will always return the singleton partition of G, which will lead to an endless loop, as G will  become
    # the aggregate graph of G with respect to the singleton partition, which is just G again.
    # Thus, 𝓟 will also be set to the value it had before, and, as we got to refine_partition, len(𝓟) != G.order() and thus, we'd get an
    # infinite loop.
    𝓟ₚ = None

    while True:
        𝓟 = move_nodes_fast(G, 𝓟, 𝓗)

        # When every community consists of a single node only, terminate, returning the flat partition given by 𝓟.
        # Also terminate, if the sequence of partition generated becomes stationary.
        if len(𝓟) == G.order() or 𝓟 == 𝓟ₚ:
            # Return the partition 𝓟 in terms of the original graph, which was passed to this function
            return 𝓟.flatten()

        # Remember partition for termination check.
        𝓟ₚ = 𝓟

        # Refine the partition created by fast local moving, potentially splitting a community into multiple parts
        𝓟ᵣ = refine_partition(G, 𝓟, 𝓗, θ, γ)
        # Create the aggregate graph of G based on 𝓟ᵣ …
        G = 𝓟ᵣ.aggregate_graph()

        # … but maintain partition 𝓟, that is, lift it to the aggregate graph.
        # The following lines are equivalent to, but way faster than
        # `partitions = [{v for v in G.nodes if G.nodes[v][Keys.NODES] <= C} for C in 𝓟]`.
        partitions: dict[int, set[T]] = {id: set() for id in range(len(𝓟))}
        # Iterate through the aggregate graph's nodes
        for v_agg, nodes in G.nodes(data=Keys.NODES):
            # Get the id of the community that the nodes collected in this super node were part of
            community_id = 𝓟._node_part[next(iter(nodes))]
            # Note that down in the partitions dict
            partitions[community_id] = partitions[community_id] | {v_agg}
        # Now, discard the indices and produce the list of values, i.e. the lifted partition
        partitions_l: list[set[T]] = list(partitions.values())

        𝓟 = Partition.from_partition(G, partitions_l, Keys.WEIGHT)


def move_nodes_fast(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T]) -> Partition[T]:
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
        # Cₘ is the optimal community, 𝛥𝓗 is the increase of 𝓗 over 𝓗ₒ, reached at Cₘ.
        (Cₘ, 𝛥𝓗, _) = argmax(lambda C: 𝓗.delta(𝓟, v, C), [*𝓟.adjacent_communities(v), set()])

        # If we can achieve a strict improvement
        if 𝛥𝓗 > 0:
            # Move node v to community Cₘ
            𝓟.move_node(v, Cₘ)

            # Identify neighbors of v that are not in Cₘ
            N = {u for u in G[v] if u not in Cₘ}

            # Visit these neighbors as well
            Q.extend(N - set(Q))

        # If queue is empty, return 𝓟
        if len(Q) == 0:
            return 𝓟


def refine_partition(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], θ: float, γ: float) -> Partition[T]:
    """Refine all communities by merging repeatedly, starting from a singleton partition."""
    # Assign each node to its own community
    𝓟ᵣ: Partition[T] = Partition.singleton_partition(G, Keys.WEIGHT)

    # Visit all communities
    for C in 𝓟:
        # refine community
        𝓟ᵣ = merge_nodes_subset(G, 𝓟ᵣ, 𝓗, θ, γ, C)

    return 𝓟ᵣ


def merge_nodes_subset(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], θ: float, γ: float, S: Set[T]) -> Partition[T]:
    """Merge the nodes in the subset S into one or more sets to refine the partition 𝓟."""
    size_s = node_total(G, S)

    R = {
        v for v in S
          if nx.cut_size(G, [v], S - {v}, weight=Keys.WEIGHT) >= γ * node_total(G, v) * (size_s - node_total(G, v))
    }  # fmt: skip

    for v in R:
        # If v is in a singleton community, i.e. is a node that has not yet been merged
        if len(𝓟.node_community(v)) == 1:
            # Consider only well-connected communities
            𝓣 = freeze([
                C for C in 𝓟
                  if C <= S and nx.cut_size(G, C, S - C, weight=Keys.WEIGHT) >= γ * float(node_total(G, C) * (size_s - node_total(G, C)))
            ])  # fmt: skip

            # Now, choose a random community to put v into
            # We use python's random.choices for the weighted choice, as this is easiest.

            # Have a list of pairs of communities in 𝓣 together with the improvement (𝛥𝓗) of moving v to the community
            # Only consider communities for which the quality function doesn't degrade, if v is moved there
            communities = [(C, 𝛥𝓗) for (C, 𝛥𝓗) in ((C, 𝓗.delta(𝓟, v, C)) for C in 𝓣) if 𝛥𝓗 >= 0]
            # Calculate the weights for the random choice using the 𝛥𝓗 values
            weights = [exp(𝛥𝓗 / θ) for (C, 𝛥𝓗) in communities]

            # Finally, choose the new community
            # Use [0][0] to extract the community, since choices returns a list, containing a single (C, 𝛥𝓗) tuple
            Cₙ = choices(communities, weights=weights, k=1)[0][0]

            # And move v there
            𝓟.move_node(v, Cₙ)

    return 𝓟
