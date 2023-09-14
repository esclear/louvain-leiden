"""This module provides some useful types and functions used in the algorithms implementations."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations_with_replacement
from typing import (  # noqa: UP035 # ruff recommends to import Callable, Iterable, Iterator from collections.abc
    Callable,
    Collection,
    Generic,
    Iterable,
    Iterator,
    TypeAlias,
    TypeVar,
)

from networkx import Graph, cut_size
from networkx.algorithms.community import community_utils

S = TypeVar("S")
Nested: TypeAlias = S | Iterable['Nested[S]']
T = TypeVar("T", covariant=True)


class Partition(Generic[T]):
    """This class represents a partition of a graph's nodes."""

    def __init__(self, G: Graph, sets: list[set[T]], node_part: dict[T, int], degree_sums: list[int], weight: None | str = None):
        """
        Create a new partition of the graph G, given by the nodes in the partition 𝓟 of G's nodes.

        This constructor is meant for internal use only, please use `Partition.from_partition` instead.
        """
        assert G.order() == len(node_part), "node_part size doesn't match number of nodes."
        assert degree_sums is not None, "No degree_sums given!"
        # Remember the graph (i.e. a reference to it)
        self.G = G

        self.graph_size = G.size(weight=weight)

        # The partition as a list of sets
        # We store /lists/ of sets instead of /sets/ of sets, because changeable sets in python are not /hashable/ and
        # thus can't be stored in a set. We could store a set of frozensets instead, however, this would complicate
        # operations such as the move_node operation below, where we modify the partitions.
        self._sets = sets

        # For faster moving of nodes, store for each node the community it belongs to.
        # This is a dict, mapping each node to its community (the community's index in self._sets).
        self._node_part = node_part

        # Store the key which is used for getting the weight information
        self._weight = weight
        self._partition_degree_sums = degree_sums

    @classmethod
    def from_partition(cls, G: Graph, 𝓟: Collection[Collection[T]] | Partition[T], weight: None | str = None) -> Partition[T]:
        """Create a new partition of the graph G, given by the nodes in the partition 𝓟 of G's nodes."""
        if not Partition.is_partition(G, 𝓟):
            raise AssertionError("𝓟 must be a partition of G!")

        # The partition as a list of sets
        # Transform the given collection into a list of sets representing the communities
        sets = [set(c) for c in 𝓟]

        # Generate the lookup table
        # (The order of nested comprehensions is a bit unintuitive in python.)
        node_part = {v: idx for idx, com in enumerate(sets) for v in com}

        partition_degree_sums = [sum(map(lambda t: t[1], G.degree(C, weight=weight))) for C in sets]

        return cls(G, sets, node_part, partition_degree_sums, weight)

    @classmethod
    def singleton_partition(cls, G: Graph, weight: None | str = None) -> Partition[T]:
        """Create a singleton partition, in which each community consists of exactly one vertex."""
        # Generate a list of triples containing all necessary information: The community (a singleton set), a (node, index) tuple
        # for the corresponding entry in the node_part lookup dict, and the degree.
        data = [({v}, (v, i), G.degree(v, weight=weight)) for i, v in enumerate(G.nodes)]
        if not data:
            # Handle the empty graphs -> return empty lists and empty lookup dict
            sets, node_part, degree_sums = [], dict(), []
        else:
            # Otherwise, split `data` into the respective representations we need to create the partition:
            # Sets becomes the list of (singleton) communities, part_tuples the list of (node, index) tuples and degree_sums a
            # list of node degrees.
            sets, part_tuples, degree_sums = map(list, zip(*data))
            # From the list of tuples, create the dictionary we need
            node_part = dict(part_tuples)

        return cls(G, sets, node_part, degree_sums, weight)

    @staticmethod
    def is_partition(G: Graph, 𝓟: Collection[Collection[T]]) -> bool:
        """Determine whether 𝓟 is indeed a partition of G."""
        # There used to be a custom implementation here, which turned out to be similar to Networkx' implementation.
        # Since I expect Networkx' implementation to be as optimized as possible and since this is only used as a
        # sanity check in the constructor, I decided to let the experts handle this.
        result: bool = community_utils.is_partition(G, 𝓟)
        return result

    @staticmethod
    def __collect_nodes(G: Graph, nodes: Collection[int | T]) -> list[T]:
        """Collect the nodes in the underlying graph that correspond to the given `nodes` in the aggregate graph `G`."""
        if "parent_partition" not in G.graph or "parent_graph" not in G.graph:
            # If none exists (i.e. we have the original graph) return G and the node we have found
            return list(nodes)
        else:
            # Otherwise, get the parent graph
            H = G.graph["parent_graph"]
            # For every node in `nodes`, collect its child nodes using recursive calls and combine them into a single list
            return sum((Partition.__collect_nodes(H, G.nodes[n]["nodes"]) for n in nodes), [])

    @staticmethod
    def __find_original_graph(G: Graph) -> tuple[Graph, list[T]]:
        """Find the original graph of an aggregate partition."""
        if "parent_graph" in G.graph:
            return Partition.__find_original_graph(G.graph["parent_graph"])
        else:
            return G

    def __copy__(self) -> Partition[T]:
        """Create a copy of this partition object."""
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.G = self.G
        cpy._sets = deepcopy(self._sets)
        cpy._node_part = self._node_part.copy()
        cpy._partition_degree_sums = self._partition_degree_sums.copy()
        cpy._weight = self._weight
        return cpy

    def __iter__(self) -> Iterator[set[T]]:
        """Make a Partition object iterable, returning an iterator over the communities."""
        return self._sets.__iter__()

    def __contains__(self, nodes: object) -> bool:
        """Return whether a given set of nodes is part of the partition or not."""
        return nodes in self._sets

    def __len__(self) -> int:
        """Get the size (number of communities) of the partition."""
        return len(self._sets)

    # In normal circumstances, using covariant type variables as function parameter (as we do here with T) can cause problems.
    # (especially for collections; see e.g. https://github.com/python/mypy/issues/7049#issuecomment-504928431 for an explanation).
    # However, ths is not a problem for move_node, as we don't add new entries to the partition and don't rely on any functionality of the
    # type T, which is only used as a type marker here.
    def move_node(self, v: T, target: set[T] | frozenset[T]) -> Partition[T]:  # type: ignore
        """Move node v from its current community in this partition to the given target community."""
        # Determine the index of the community that v was in initially
        source_partition_idx = self._node_part[v]

        # If the target set is non-empty, i.e. an existing community, determine its index in _sets
        if len(target) > 0:
            # Get any element of the target set …
            el = next(iter(target))
            # … and query its index in the _sets list
            target_partition_idx = self._node_part[el]
        # Otherwise, create a new (currently empty) partition and get its index.
        else:
            target_partition_idx = len(self._sets)
            self._sets.append(set())
            self._partition_degree_sums.append(0)

        # Remove `v` from its old community and place it into the target partition
        self._sets[source_partition_idx].discard(v)
        self._sets[target_partition_idx].add(v)
        # Also update the sum of node degrees in that partition
        deg_v = self.G.degree(v, weight=self._weight)
        self._partition_degree_sums[source_partition_idx] -= deg_v
        self._partition_degree_sums[target_partition_idx] += deg_v

        # Update v's entry in the index lookup table
        self._node_part[v] = target_partition_idx

        # If the original partition is empty now, that we removed v from it, remove it and adjust the indexes in _node_part
        if len(self._sets[source_partition_idx]) == 0:
            # Remove the now empty set from `self._sets`
            self._sets.pop(source_partition_idx)
            self._partition_degree_sums.pop(source_partition_idx)
            # And adjust the indices in the lookup table
            self._node_part = {v: (i if i < source_partition_idx else i - 1) for v, i in self._node_part.items()}

        return self

    # We ignore the typing check for the following function, as it is only a read-only function:
    # Using a covariant type variable as a function parameter (as we do here with T) can cause problems.
    # (see e.g. https://github.com/python/mypy/issues/7049#issuecomment-504928431 for an explanation).
    # However, as node_community serves as a pure read-only function, doing so poses no problem here and keeps the API simple.
    def node_community(self, v: T) -> set[T]:  # type: ignore
        """Get the community the node v is currently part of."""
        return self._sets[self._node_part[v]]

    def as_set(self) -> set[frozenset[T]]:
        """Return a set of sets of nodes that represents the communities."""
        return freeze(self.communities)

    # Here, we also permit a covariant type variable as a function parameter, as this is a pure read-only function (c.f. node_community).
    def degree_sum(self, v: T) -> int:  # type: ignore
        """Get the sum of node degrees of nodes in the community that `v` belongs to."""
        return self._partition_degree_sums[self._node_part[v]]

    def flatten(self) -> Partition[T]:
        """Flatten the partition, producing a partition of the original graph."""
        # If this is not an aggregate graph, return self.
        if "parent_graph" not in self.G.graph or "parent_partition" not in self.G.graph:
            return self

        # Otherwise
        G: Graph = Partition.__find_original_graph(self.G)
        𝓟 = [Partition.__collect_nodes(self.G, C) for C in self._sets]

        return Partition.from_partition(G, 𝓟)

    @property
    def communities(self) -> tuple[set[T], ...]:
        """
        Return the communities in this partition as a tuple.

        We're using tuples as an immutable representation of a set / list, that is, the order of entries is of no importance.
        """
        return tuple(self._sets)


def freeze(set_list: Iterable[set[T] | frozenset[T]]) -> set[frozenset[T]]:
    """
    Given a list of set, return a set of (frozen) sets representing those sets.

    This function returns a set of *frozen* sets, as plain sets are not hashable
    in python and thus cannot be contained in a set.
    """
    return set(map(lambda c: frozenset(c), set_list))


def node_total(G: Graph, S: T | Collection[T]) -> int:
    """Return the recursive size of the set S."""
    if not isinstance(S, Iterable):
        # TODO: Weight parameter name hardcoded!
        return G.nodes.data("weight", default=1)[S]
    else:
        return sum(node_total(G, v) for v in S)


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


def aggregate_graph(G: Graph, 𝓟: Partition[T], weight: str | None = None) -> Graph:
    """
    Create an aggregate graph of the graph G with regards to the partition 𝓟.

    The aggregate graph is a multigraph, in which the nodes of every partition set have been coalesced into a single
    node. Every edge between two nodes a and b is represented by an edge in the multigraph, between the nodes that
    represent the communities that a and b, respectively, are members of.
    """
    # Determine the numer of communities and get a list of the communities
    n_c = len(𝓟)
    communities = list(𝓟.communities)
    node_weights = G.nodes.data(weight, default=1)

    # Create graph H that will become the aggregate graph
    H = Graph(parent_graph=G, parent_partition=𝓟)

    # For every community, add a node in H, also recording the nodes
    for i, C in enumerate(communities):
        community_weight = sum(node_weights[v] for v in C)
        H.add_node(i, weight=community_weight, nodes=frozenset(C))

    # For every pair of communities, determine the total weight of edges between them.
    # This also includes edges between two nodes in the same community, which will form a loop in the aggregate graph.
    for c_idx, d_idx in combinations_with_replacement(range(n_c), 2):
        C, D = communities[c_idx], communities[d_idx]
        H.add_edge(c_idx, d_idx, weight=cut_size(G, C, D, weight=weight))

    return H
