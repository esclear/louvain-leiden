"""This module provides some useful types and functions used in the algorithms implementations."""
from __future__ import annotations

from functools import reduce
from typing import Callable, TypeVar  # noqa: UP035 # recommends to import Callable from collections.abc instead

from networkx import Graph, MultiGraph
from networkx.algorithms.community import community_utils

T = TypeVar("T")

class Partition:
    """This class represents a partition of a graph's nodes."""

    T = TypeVar("T")

    def __init__(self, G: Graph, P: list[set[T]]):
        """Create a new partition of the graph G, given by the nodes in the partition P of G's nodes."""
        assert Partition.is_partition(G, P), "P must be a partition of G!"

        # Remember the graph
        self.G = G

        # The partition as a list of sets
        # We store /lists/ of sets instead of /sets/ of sets, because changeable sets in python are not /hashable/ and
        # thus can't be stored in a set. We could store a set of frozensets instead, however, this would complicate
        # operations such as the move_node operation below, where we modify the partitions.
        self._sets = P

        # For faster moving of nodes, store for each node the community it belongs to
        # The result is a dict that maps a node to its community (a set of nodes, containing that node).
        self._node_part = {
            v: c for c in P for v in c
        }  # the order is a bit unintuitive in python

    @staticmethod
    def is_partition(G: Graph, 𝓟: list[set[T]]) -> bool:
        """Determine whether 𝓟 is indeed a partition of G."""
        # There used to be a custom implementation here, which turned out to be similar to Networkx' implementation.
        # Since I expect Networkx' implementation to be as optimized as possible and since this is only used as a
        # sanity check in the constructor, I decided to let the experts handle this.
        return community_utils.is_partition(G, 𝓟)

    def move_node(self, v: T, target: set[T]) -> Partition:
        """Move node v from its current community in this partition to the given target community."""
        # Sanity check: the target community is indeed a community in this partition
        assert target in self._sets or target == {}

        new_partitions = [
            # Add v to the target community and remove v from all other communities …
            # (removing v only from its previous community in practice.)
            (p | {v} if p == target else p - {v})
            # … p in this parition.
            for p in self
        ] + (
            [{v}] if target == {} else []
        )  # If the target is an empty set, also include v, otherwise don't

        # And remove empty sets from the partition
        new_partitions = [p for p in new_partitions if len(p) > 0]

        return Partition(self.G, new_partitions)

    def node_community(self, v: T) -> set[T]:
        """Get the community the node v is currently part of."""
        return self._node_part[v]

    def __iter__(self):
        """Make a Partition object iterable, returning an iterator over the communities."""
        return self._sets.__iter__()

    def as_set(self) -> set[set[T]]:
        """Return a set of sets of nodes that represents the communities."""
        return freeze(self._sets)

    @property
    def size(self):
        """Gets the size (number of communities) of the partition."""
        return len(self._sets)

    @property
    def communities(self):
        """
        Return the communities in this partition as a tuple.

        The order is of no importance; we're using tuples as an immutable representation of a set / list.
        """
        return tuple(self._sets)


def freeze(set_list: list[set[T]]) -> set[set[T]]:
    """
    Given a list of set, return a set of (frozen) sets representing those sets.

    This function returns a set of *frozen* sets, as plain sets are not hashable
    in python and thus cannot be contained in a set.
    """
    return set(map(lambda c: frozenset(c), set_list))


def recursive_size(S: list | object) -> int:
    """Return the recursive size of the set S."""
    if not isinstance(S, list):
        return 1

    return sum(recursive_size(s) for s in S)


def flat(S: set | object) -> set:
    """Flatten potentially nested sets into a flattened (non-nested) set."""
    # "unfreeze" frozen sets
    if isinstance(S, frozenset):
        S = set(S)

    if not isinstance(S, set):
        return {S}

    return reduce(lambda a, s: a | s, (flat(s) for s in S), set())


def flatₚ(𝓟: Partition) -> list[set[T]]:
    """
    Flatten a partition into a *list* of communities (each of which represented as a set).

    This is used for partitions of aggregate graphs, where multiple nodes have been
    coalesced into one single node, which is represented by the set of the original nodes.
    """
    return [flat(C) for C in 𝓟]


def argmax(
    objective_function: Callable[[T], float], parameters: list[T]
) -> tuple[T, float, int] | None:
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

    # find the maximum by iterating over the remaining indices (beginning at index 1)
    for k in range(1, len(parameters)):
        optₖ = parameters[k]
        valₖ = objective_function(optₖ)

        if valₖ > val:
            idx = k
            opt = optₖ
            val = valₖ

    return (opt, val, idx)


def aggregate_graph(G: Graph, 𝓟: Partition) -> MultiGraph:
    """
    Create an aggregate graph of the graph G with regards to the partition 𝓟.

    The aggregate graph is a multigraph, in which the nodes of every partition set have been coalesced into a single
    node. Every edge between two nodes a and b is represented by an edge in the multigraph, between the nodes that
    represent the communities that a and b, respectively, are members of.
    """
    H = MultiGraph()
    H.add_nodes_from([frozenset(c) for c in 𝓟])

    for (u, v) in G.edges():
        C = frozenset(𝓟.node_community(u))
        D = frozenset(𝓟.node_community(v))

        H.add_edge(C, D)

    return H


def singleton_partition(G: Graph) -> Partition:
    """Create a singleton partition, in which each community consists of exactly one vertex."""
    # Partition as list of sets
    P = [{v} for v in G.nodes]
    return Partition(G, P)
