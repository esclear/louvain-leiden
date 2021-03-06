from networkx import Graph, MultiGraph

from utils import *


def move_nodes(G: Graph, π: Partition, π: QualityFunction) -> Partition:
    while True:
        # Store current ("old") quality function value
        πβ = π(G, π)

        for v in G.nodes:
            # Find best community for node `v` to be in, potentially creating a new community.
            # Cβ is the optimal community, π₯π is the increase of π over πβ, reached at Cβ.
            (Cβ, π₯π, _) = argmax(lambda C: π(G, π.move_node(v, C)) - πβ, π.sets + [{}])

            # If we get a strictly better value, assign v to community Cβ
            if π₯π > 0:
                π = π.move_node(v, Cβ)

        # If no further improvement can be made, we're done and return the current partition
        if π(G, π) <= πβ:
            return π


def louvain(G: Graph, π: QualityFunction, π: Partition = None) -> Partition:
    """
    Implementation of the Louvain algorithm for community detection.
    """
    # If there is no partition given, start with all nodes in the same community
    if π == None:
        π = Partition(G, [{v for v in G.nodes}])

    # Remember the original graph
    O = G
    while True:
        π = move_nodes(G, π, π)

        # When every community consists of a single node only, terminate,
        # returning the flat partition given by π
        if len(π.sets) == len(G.nodes):
            return Partition(O, flatβ(π))

        # Create the aggregate graph of G based on π
        G = aggregate_graph(G, π)
        # And update π to be a singleton partition of G
        π = singleton_partition(G)
