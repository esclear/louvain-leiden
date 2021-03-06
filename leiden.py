from math import exp
from random import choices
from typing import Set

from networkx import Graph, MultiGraph

from utils import *


def leiden(G: Graph, π: QualityFunction, π: Partition = None, ΞΈ: float = 2.0, Ξ³: float = 3.0) -> Partition:
    """
    Implementation of the Leiden algorithm for community detection.
    """
    # If there is no partition given, start with all nodes in the same community
    if π == None:
        π = Partition(G, [{v for v in G.nodes}])

    # Remember the original graph
    O = G
    while True:
        π = move_nodes_fast(G, π, π)

        # When every community consists of a single node only, terminate,
        # returning the flat partition given by π
        if len(π.sets) == len(G.nodes):
            return Partition(O, flatβ(π))

        πα΅£ = refine_partition(G, π, π, ΞΈ, Ξ³)
        # Create the aggregate graph of G based on πα΅£ β¦
        G = aggregate_graph(G, πα΅£)

        # β¦ but maintain partition π
        π = Partition(G, [{v for v in G.nodes if v <= C} for C in P])


def move_nodes_fast(G: Graph, π: Partition, π: QualityFunction) -> Partition:
    # Create a queue of all nodes to visit them in random order
    Q = set(G.nodes)
    
    while True:
        # Store current ("old") quality function value
        πβ = π(G, π)

        # Determine next node to visit
        v = Q.pop()

        # Find best community for node `v` to be in, potentially creating a new community.
        # Cβ is the optimal community, π₯π is the increase of π over πβ, reached at Cβ.
        (Cβ, π₯π, _) = argmax(lambda C: π(G, π.move_node(v, C)) - πβ, π.sets + [{}])

        # If we can achieve a strict improvement
        if π₯π > 0:
            # Move node v to community Cβ
            π = π.move_node(v, Cβ)

            # Identify neighbors of v that are not in Cβ
            N = {u for u in G.neighbors(v) if u not in Cβ}

            # Visit these neighbors next
            Q.update(N - Q)

        # If queue is empty, return π
        if len(Q) == 0:
            return π


def refine_partition(G: Graph, π: Partition, π: QualityFunction, ΞΈ: float, Ξ³: float) -> Partition:
    # Assign each node to its own community
    πα΅£ = singleton_partition(G)

    # Visit all communities
    for C in π:
        # refine community
        πα΅£ = merge_nodes_subset(G, πα΅£, π, ΞΈ, Ξ³, C)

    return πα΅£


def merge_nodes_subset(G: Graph, π: Partition, π: QualityFunction, ΞΈ: float, Ξ³: float, S: Set[T]) -> Partition:
    R = { v for v in S if len(G.edges(v, frozenset(S - {v}))) >= Ξ³ * recursive_size(v) * (recursive_size(S) - recursive_size(v)) }
    
    for v in R:
        # If v is in a singleton community, i.e. is a node that has not yet been merged
        if len(π.node_community(v)) == 1:
            # Consider only well-connected communities
            π£ = { frozenset(C) for C in π if C <= S and len(G.edges(C, frozenset(S - C))) >= Ξ³ * recursive_size(C) * (recursive_size(S) - recursive_size(C)) }

            # Now, choose a random community to put v into
            # We use python's random.choice for the weighted choice, as this is easiest.

            # Store current ("old") quality function value
            πβ = π(G, π)

            # Communities with the improvement (π₯π) of moving v there
            communities = [ (C, π(G, π.move_node(v, C)) - πβ) for C in π£ ]
            # Only consider communities for which the quality function doesn't degrade, if v is moved there
            communities = list(filter(lambda C_π₯π: C_π₯π[1] >= 0, communities)) # Python 3 removed the option to destructure tuples that are lambda arguments :(

            weights = [ exp(π₯π / ΞΈ) for (C, π₯π) in communities ]

            # Finally, choose the new community
            # Use [0][0] to extract the community, since choices returns a list, containing a single (C, π₯π) tuple
            Cβ = choices(communities, weights)[0][0]

            # And move v there
            π = π.move_node(v, Cβ)

    return π
