"""This module provides some useful types and functions used in the algorithms implementations."""
from __future__ import annotations

import copy
from functools import reduce
from typing import (  # noqa: UP035 # recommends to import Callable from collections.abc instead
    Callable,
    Collection,
    Generic,
    Iterable,
    Iterator,
    TypeAlias,
    TypeVar,
    cast,
)

from networkx import Graph, MultiGraph
from networkx.algorithms.community import community_utils

S = TypeVar("S")
Nested: TypeAlias = S | Iterable['Nested[S]']
T = TypeVar("T", covariant=True)


class Partition(Generic[T]):
    """This class represents a partition of a graph's nodes."""

    def __init__(self, G: Graph, sets: list[set[T]], node_part: dict[T, int]):
        """
        Create a new partition of the graph G, given by the nodes in the partition 𝓟 of G's nodes.

        This constructor is meant for internal use only, please use `Partition.from_partition` instead.
        """
        # Remember the graph (i.e. a reference to it)
        self.G = G

        # The partition as a list of sets
        # We store /lists/ of sets instead of /sets/ of sets, because changeable sets in python are not /hashable/ and
        # thus can't be stored in a set. We could store a set of frozensets instead, however, this would complicate
        # operations such as the move_node operation below, where we modify the partitions.
        self._sets = sets

        # For faster moving of nodes, store for each node the community it belongs to.
        # This is a dict, mapping each node to its community (the community's index in self._sets).
        self._node_part = node_part

    @classmethod
    def from_partition(cls, G: Graph, 𝓟: Collection[Collection[T]] | Partition[T]) -> Partition[T]:
        """Create a new partition of the graph G, given by the nodes in the partition 𝓟 of G's nodes."""
        if not Partition.is_partition(G, 𝓟):
            raise AssertionError("𝓟 must be a partition of G!")

        # The partition as a list of sets
        # Transform the given collection into a list of sets representing the communities
        sets = [set(c) for c in 𝓟]

        # Generate the lookup table
        # (The order of nested comprehensions is a bit unintuitive in python.)
        node_part = {v: idx for idx, com in enumerate(sets) for v in com}

        return cls(G, sets, node_part)

    @staticmethod
    def is_partition(G: Graph, 𝓟: Collection[Collection[T]]) -> bool:
        """Determine whether 𝓟 is indeed a partition of G."""
        # There used to be a custom implementation here, which turned out to be similar to Networkx' implementation.
        # Since I expect Networkx' implementation to be as optimized as possible and since this is only used as a
        # sanity check in the constructor, I decided to let the experts handle this.
        result: bool = community_utils.is_partition(G, 𝓟)
        return result

    # In normal circumstances, using covariant type variables as function parameter (as we do here with T) can cause problems.
    # (see e.g. https://github.com/python/mypy/issues/7049#issuecomment-504928431 for an explanation).
    # However, ths is not a problem for move_node and node_community, as the type variable T is only used as a type marker and
    # we don't rely on *any* functionality of T at all. Thus, to keep the user interface easy to use, we ignore the type check
    # for the signatures of move_node and node_community down below.
    def move_node(self, v: T, target: set[T] | frozenset[T]) -> Partition[T]:  # type: ignore
        """Move node v from its current community in this partition to the given target community."""
        sets, node_part = copy.deepcopy(self._sets), self._node_part.copy()
        # Determine the index of the community that v was in initially
        source_partition_idx = node_part[v]

        # If the target set is non-empty, i.e. an existing community, determine its index in _sets
        if len(target) > 0:
            # Get any element of the target set …
            el = next(iter(target))
            # … and query its index in the _sets list
            target_partition_idx = node_part[el]
        # Otherwise, create a new (currently empty) partition and get its index.
        else:
            target_partition_idx = len(sets)
            sets.append(set())

        # Remove `v` from its old community and place it into the target partition
        sets[source_partition_idx].discard(v)
        sets[target_partition_idx].add(v)

        # Update v's entry in the index lookup table
        node_part[v] = target_partition_idx

        # If the original partition is empty now, that we removed v from it, remove it and adjust the indexes in _node_part
        if len(sets[source_partition_idx]) == 0:
            # Remove the now empty set from `sets`
            sets.pop(source_partition_idx)
            # And adjust the indices in the lookup table
            node_part = {v: (i if i < source_partition_idx else i - 1) for v, i in node_part.items()}

        return Partition(self.G, sets, node_part)

    # We ignore the typing check for the following function, as it is only a read-only function:
    # Using a covariant type variable as a function parameter (as we do here with T) can cause problems.
    # (see e.g. https://github.com/python/mypy/issues/7049#issuecomment-504928431 for an explanation).
    # However, as node_community serves as a pure read-only function, doing so poses no problem here and keeps the API simple.
    def node_community(self, v: T) -> set[T]:  # type: ignore
        """Get the community the node v is currently part of."""
        return self._sets[self._node_part[v]]

    def __iter__(self) -> Iterator[set[T]]:
        """Make a Partition object iterable, returning an iterator over the communities."""
        return filter(lambda s: len(s) > 0, self._sets)

    def __contains__(self, nodes: object) -> bool:
        """Return whether a given set of nodes is part of the partition or not."""
        return nodes in self._sets

    def as_set(self) -> set[frozenset[T]]:
        """Return a set of sets of nodes that represents the communities."""
        return freeze(self.communities)

    def __len__(self) -> int:
        """Get the size (number of communities) of the partition."""
        return len(self._sets)

    @property
    def communities(self) -> tuple[set[T], ...]:
        """
        Return the communities in this partition as a tuple.

        We're using tuples as an immutable representation of a set / list, that is, the order of entries is of no importance.
        """
        return tuple(filter(lambda s: len(s) > 0, self._sets))


def freeze(set_list: Iterable[set[T] | frozenset[T]]) -> set[frozenset[T]]:
    """
    Given a list of set, return a set of (frozen) sets representing those sets.

    This function returns a set of *frozen* sets, as plain sets are not hashable
    in python and thus cannot be contained in a set.
    """
    return set(map(lambda c: frozenset(c), set_list))


def recursive_size(S: Nested[T]) -> int:
    """Return the recursive size of the set S."""
    if isinstance(S, list):
        return sum(recursive_size(s) for s in cast(list[Nested[T]], S))

    return 1


def flat(S: Nested[T]) -> set[T]:
    """Flatten potentially nested sets into a flattened (non-nested) set."""
    if isinstance(S, Iterable):
        return reduce(lambda a, s: a | s, (flat(s) for s in S), set())

    return {S}


def flatₚ(𝓟: Partition[Nested[T]]) -> list[set[T]]:
    """
    Flatten a partition into a *list* of communities (each of which represented as a set).

    This is used for partitions of aggregate graphs, where multiple nodes have been
    coalesced into one single node, which is represented by the set of the original nodes.
    """
    return [flat(C) for C in 𝓟]


def argmax(objective_function: Callable[[T], float], parameters: list[T]) -> tuple[T, float, int]:
    """
    Find the arg max with respect to a given objective function over a given list of parameters.

    If parameters is None or empty, the result will be None.
    Otherwise, the first item of the result will be the arg max, the second item will be the maximum
    and the third parameter will be the index of the maximum in the `parameters` list.
    """
    if not parameters:
        raise ValueError("The given `parameters` must be a non-empty list!")

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


def aggregate_graph(G: Graph, 𝓟: Partition[T]) -> MultiGraph:
    """
    Create an aggregate graph of the graph G with regards to the partition 𝓟.

    The aggregate graph is a multigraph, in which the nodes of every partition set have been coalesced into a single
    node. Every edge between two nodes a and b is represented by an edge in the multigraph, between the nodes that
    represent the communities that a and b, respectively, are members of.
    """
    H = MultiGraph()
    H.add_nodes_from([frozenset(C) for C in 𝓟])

    for u, v in G.edges():
        C = frozenset(𝓟.node_community(u))
        D = frozenset(𝓟.node_community(v))

        H.add_edge(C, D)

    return H


def singleton_partition(G: Graph) -> Partition[T]:
    """Create a singleton partition, in which each community consists of exactly one vertex."""
    # Partition as list of sets
    𝓟 = [{v} for v in G.nodes]
    return Partition.from_partition(G, 𝓟)
