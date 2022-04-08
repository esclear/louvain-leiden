from networkx import Graph, MultiGraph

from utils import *


def move_nodes(G: Graph, 𝓟: Partition, 𝓗: QualityFunction) -> Partition:
    while True:
        # Store current ("old") quality function value
        𝓗ₒ = 𝓗(G, 𝓟)

        for v in G.nodes:
            # Find best community for node `v` to be in, potentially creating a new community.
            # Cₘ is the optimal community, 𝛥𝓗 is the increase of 𝓗 over 𝓗ₒ, reached at Cₘ.
            (Cₘ, 𝛥𝓗, _) = argmax(lambda C: 𝓗(G, 𝓟.move_node(v, C)) - 𝓗ₒ, 𝓟.sets + [{}])

            # If we get a strictly better value, assign v to community Cₘ
            if 𝛥𝓗 > 0:
                𝓟 = 𝓟.move_node(v, Cₘ)

        # If no further improvement can be made, we're done and return the current partition
        if 𝓗(G, 𝓟) <= 𝓗ₒ:
            return 𝓟


def louvain(G: Graph, 𝓗: QualityFunction, 𝓟: Partition = None) -> Partition:
    """
    Implementation of the Louvain algorithm for community detection.
    """
    # If there is no partition given, start with all nodes in the same community
    if 𝓟 == None:
        𝓟 = Partition(G, [{v for v in G.nodes}])

    # Remember the original graph
    O = G
    while True:
        𝓟 = move_nodes(G, 𝓟, 𝓗)

        # When every community consists of a single node only, terminate,
        # returning the flat partition given by 𝓟
        if len(𝓟.sets) == len(G.nodes):
            return Partition(O, flatₚ(𝓟))

        # Create the aggregate graph of G based on 𝓟
        G = aggregate_graph(G, 𝓟)
        # And update 𝓟 to be a singleton partition of G
        𝓟 = singleton_partition(G)
