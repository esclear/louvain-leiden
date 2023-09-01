"""
Implementation of the Leiden algorithm for community detection.

This implementation follows the outline provided in the supplementary material of the paper "From Louvain to Leiden:
guaranteeing well-connected communities" by V.A. Traag, L. Waltman and N.J. van Eck.
"""

from math import exp
from random import choices, shuffle
from typing import TypeVar

from networkx import Graph, edge_boundary

from .quality_metrics import QualityMetric
from .utils import Partition, aggregate_graph, argmax, flatₚ, freeze, recursive_size

T = TypeVar("T")


def leiden(G: Graph, 𝓗: QualityMetric[T], 𝓟: Partition[T] | None = None, θ: float = 0.05, γ: float = 1.0) -> Partition[T]:
    """
    Perform the Leiden algorithm for community detection.

    Parameters
    ----------
    G : Graph
        The graph / network to process
    𝓗 : QualityMetric[T]
        A quality metric to optimize
    𝓟 : Partition[T], optional
        A partition to refine, leave at the default of `None` when not refining an existing partition.
    θ : float, optional
        The θ parameter of the Leiden method, which determines the randomness in the refinement phase of the Leiden
        algorithm, default value of 0.05
    γ : float, optional
        The γ parameter of the Leiden method, default value of 3.0

    :returns: A partition of G into communities
    """
    # If there is no partition given, start with all nodes in the same community
    if 𝓟 is None:
        𝓟 = Partition.from_partition(G, [{v for v in G.nodes}])

    # Remember the original graph
    G_orig = G
    while True:
        𝓟 = move_nodes_fast(G, 𝓟, 𝓗)

        # When every community consists of a single node only, terminate, returning the flat partition given by 𝓟
        if len(𝓟) == len(G.nodes):
            # Return the partition 𝓟 in terms of the original graph, G_orig
            return Partition.from_partition(G_orig, flatₚ(𝓟))

        𝓟ᵣ = refine_partition(G, 𝓟, 𝓗, θ, γ)
        # Create the aggregate graph of G based on 𝓟ᵣ …
        G = aggregate_graph(G, 𝓟ᵣ)

        # … but maintain partition 𝓟
        𝓟 = Partition.from_partition(G, [{v for v in G.nodes if v <= C} for C in 𝓟])


def move_nodes_fast(G: Graph, 𝓟: Partition[T], 𝓗: QualityMetric[T]) -> Partition[T]:
    """
    Perform fast local node moves to communities to improve the partition's quality.

    For every node, greedily move it to a neighboring community, maximizing the improvement in the partition's quality.
    """
    # Create a queue to visit all nodes in random order.
    Q = list(G.nodes)
    shuffle(Q)

    while True:
        # Store current ("old") quality function value
        𝓗ₒ = 𝓗(G, 𝓟)

        # Determine next node to visit by popping first node in the queue
        v = Q.pop(0)

        # Find best community for node `v` to be in, potentially creating a new community.
        # Cₘ is the optimal community, 𝛥𝓗 is the increase of 𝓗 over 𝓗ₒ, reached at Cₘ.
        (Cₘ, 𝛥𝓗, _) = argmax(lambda C: 𝓗.delta(G, 𝓟, v, C, 𝓗ₒ), [*𝓟, set()])

        # If we can achieve a strict improvement
        if 𝛥𝓗 > 0:
            # Move node v to community Cₘ
            𝓟 = 𝓟.move_node(v, Cₘ)

            # Identify neighbors of v that are not in Cₘ
            N = {u for u in G.neighbors(v) if u not in Cₘ}

            # Visit these neighbors as well
            Q.extend(N - set(Q))

        # If queue is empty, return 𝓟
        if len(Q) == 0:
            return 𝓟


def refine_partition(G: Graph, 𝓟: Partition[T], 𝓗: QualityMetric[T], θ: float, γ: float) -> Partition[T]:
    """Refine all communities by merging repeatedly, starting from a singleton partition."""
    # Assign each node to its own community
    𝓟ᵣ: Partition[T] = Partition.singleton_partition(G)

    # Visit all communities
    for C in 𝓟:
        # refine community
        𝓟ᵣ = merge_nodes_subset(G, 𝓟ᵣ, 𝓗, θ, γ, C)

    return 𝓟ᵣ


def merge_nodes_subset(G: Graph, 𝓟: Partition[T], 𝓗: QualityMetric[T], θ: float, γ: float, S: set[T] | frozenset[T]) -> Partition[T]:
    """Merge the nodes in the subset S into one or more sets to refine the partition 𝓟."""

    def E(C: set['T'] | frozenset['T'], D: set['T'] | frozenset['T']) -> int:
        """Calculate |{ (u,v) ∈ E(G) | u ∈ C, v ∈ D }|."""  # noqa: D402 # disable warning that dislikes 'E' here
        # edge_boundary (from NetworkX) calculates a C-D-cut, i.e. all edges starting in C and ending in D
        return sum(1 for _ in edge_boundary(G, C, D))

    R = {
        v for v in S
          if E({v}, S - {v}) >= γ * recursive_size(v) * (recursive_size(S) - recursive_size(v))
    }  # fmt: skip

    for v in R:
        # If v is in a singleton community, i.e. is a node that has not yet been merged
        if len(𝓟.node_community(v)) == 1:
            # Consider only well-connected communities
            𝓣 = freeze([
                C for C in 𝓟
                  if C <= S and E(C, S - C) >= γ * float(recursive_size(C) * (recursive_size(S) - recursive_size(C)))
            ])  # fmt: skip

            # Now, choose a random community to put v into
            # We use python's random.choices for the weighted choice, as this is easiest.

            # Store current ("old") quality function value
            𝓗ₒ = 𝓗(G, 𝓟)

            # Have a list of pairs of communities in 𝓣 together with the improvement (𝛥𝓗) of moving v to the community
            # Only consider communities for which the quality function doesn't degrade, if v is moved there
            communities = [(C, 𝛥𝓗) for (C, 𝛥𝓗) in ((C, 𝓗.delta(G, 𝓟, v, C, 𝓗ₒ)) for C in 𝓣) if 𝛥𝓗 >= 0]

            weights = [exp(𝛥𝓗 / θ) for (C, 𝛥𝓗) in communities]

            # Finally, choose the new community
            # Use [0][0] to extract the community, since choices returns a list, containing a single (C, 𝛥𝓗) tuple
            Cₙ = choices(communities, weights=weights, k=1)[0][0]

            # And move v there
            𝓟 = 𝓟.move_node(v, Cₙ)

    return 𝓟
