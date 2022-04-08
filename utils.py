from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, List, Optional, Set, Tuple, TypeVar, Union

from networkx import Graph, MultiGraph
from networkx.algorithms.community import community_utils


class Partition:
    T = TypeVar("T")

    def __init__(self, G: Graph, P: List[Set[T]]):
        assert Partition.is_partition(G, P), "P must be a partition of G!"

        # Remember the graph
        self.G = G

        # The partition as a list of sets
        # We store /lists/ of sets instead of /sets/ of sets, because changeable sets in python
        # are not /hashable/ and thus can't be stored in a set. We could store a set of frozensets instead,
        # however, this would complicate operations such as the move_node operation below, where we modify the partitions.
        self.sets = P

        # For faster moving of nodes, store for each node the community it belongs to
        # The result is a dict that maps a node to its community (a set of nodes, containing that node).
        self._node_part = {
            v: c for c in P for v in c
        }  # the order is a bit unintuitive in python

    @staticmethod
    def is_partition(G: Graph, 𝓟: List[Set[T]]) -> bool:
        """
        Determine whether 𝓟 is indeed a partition of G.
        """
        # There used to be a custom implementation here, which turned out to be fairly similar to Networkx' implementation.
        # Since I expect Networkx' implementation to be as optimized as possible and since this is only used as a sort of
        # sanity check in the constructor, I decided to let the experts handle this.
        return community_utils.is_partition(G, 𝓟)

    def move_node(self, v: T, target: Set[T]) -> Partition:
        """
        Move node v from its current community in this partition to the given target community.
        """
        # Sanity check: the target community is indeed a community in this partition
        assert target in self.sets or target == {}

        new_partitions = [
            # Add v to the target community and remove v from all other communities
            # (removing v only from its previous community in practice.)
            (p | {v} if p == target else p - {v})
            # for all sets p in this parition
            for p in self
        ] + ([ {v} ] if target == {} else []) # If the target is an empty set, put v in there

        # And remove empty sets from the partition
        new_partitions = [p for p in new_partitions if len(p) > 0]

        return Partition(self.G, new_partitions)

    def node_community(self, v: T) -> Set[T]:
        """
        Get the community the node v is currently part of.
        """
        return self._node_part[v]

    def __iter__(self):
        """
        Make a Partition object iterable, returning an iterator over the communities.
        """
        return self.sets.__iter__()


class QualityFunction(ABC):
    """"""

    @classmethod
    @abstractmethod
    def __call__(self, G: Graph, 𝓟: Partition) -> float:
        raise NotImplementedError()


class Modularity(QualityFunction):
    def __init__(self, γ: float = 0.25):
        self.γ = γ

    def __call__(self, G: Graph, 𝓟: Partition) -> float:
        communities = 𝓟.sets

        degree = dict(G.degree())
        deg_sum = sum(degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum ** 2

        def community_contribution(community):
            comm = set(community)
            L_c = sum(1 for u, v in G.edges(comm) if v in comm)

            degree_sum = sum(degree[u] for u in comm)

            return L_c / m - self.γ * degree_sum * degree_sum * norm

        return sum(map(community_contribution, communities))


class CPM(QualityFunction):
    """
    Implementation of the Constant Potts Model (CPM) as a quality function.
    """

    @classmethod
    def __call__(self, G: Graph, 𝓟: Partition) -> float:
        pass


def recursive_size(S: Union[List, object]) -> int:
    """
    Return the recursive size of a set S.
    """
    if not isinstance(S, list):
        return 1

    return sum(recursive_size(s) for s in S)


def flat(S: Union[Set, object]) -> Set:
    # "unfreeze" up frozen sets
    if isinstance(S, frozenset):
        S = set(S)

    if not isinstance(S, set):
        return {S}

    return reduce(lambda a, s: a | s, (flat(s) for s in S), set())


def flatₚ(𝓟: Partition) -> List:
    """
    Flatten a partition.
    """
    return [ flat(C) for C in 𝓟 ]


T = TypeVar("T")


def argmax(
    objective_function: Callable[[T], float], parameters: List[T]
) -> Optional[Tuple[T, float, int]]:
    """
    Find the arg max with respect to a given objective function over a given list of parameters.

    If parameters is None or empty, the result will be None.
    Otherwise, the first item of the result will be the arg max, the second item will be the maximum
    and the third parameter will be the index of the maximum in the `parameters` list.
    """
    if not parameters:
        return None

    idx = 0
    opt = parameters[idx]
    val = objective_function(opt)

    # find the maximum by iterating over the remaining indices (beginning at 1)
    for k in range(1, len(parameters)):
        optₖ = parameters[k]
        valₖ = objective_function(optₖ)

        if valₖ > val:
            idx = k
            opt = optₖ
            val = valₖ

    return (opt, val, idx)


def aggregate_graph(G: Graph, 𝓟: Partition) -> MultiGraph:
    H = MultiGraph()
    H.add_nodes_from([frozenset(c) for c in 𝓟])

    for (u, v) in G.edges():
        C = frozenset(𝓟.node_community(u))
        D = frozenset(𝓟.node_community(v))

        H.add_edge(C, D)

    return H


def singleton_partition(G: Graph) -> Partition:
    """
    Create a singleton partition, in which each community consists of exactly one vertex.
    """
    # Partition as list of sets
    P = [{v} for v in G.nodes]
    return Partition(G, P)
