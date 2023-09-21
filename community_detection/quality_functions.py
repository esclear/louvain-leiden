"""This module defines quality functions and provides implementations of Modularity and the ConstantPotts Model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set
from copy import copy
from math import comb
from typing import Generic, TypeVar

import networkx as nx

from .utils import Partition, single_node_neighbor_cut_size

T = TypeVar("T")


class QualityFunction(ABC, Generic[T]):
    """A metric that, when called, measures the quality of a partition into communities."""

    @abstractmethod
    def __call__(self, P: Partition[T]) -> float:
        """Measure the quality of the given partition as applied to the graph provided."""
        raise NotImplementedError()

    @abstractmethod
    def delta(self, P: Partition[T], v: T, target: Set[T]) -> float:
        """Measure the increase (or decrease, if negative) of this quality function when moving node v into the target community."""
        moved = copy(P).move_node(v, target)
        return self(moved) - self(P)


class Modularity(QualityFunction[T], Generic[T]):
    """Implementation of Modularity as a quality function."""

    def __init__(self, gamma: float = 0.25) -> None:
        """Create a new instance of Modularity quality function with the given resolution parameter gamma."""
        self.gamma: float = gamma

    def __call__(self, P: Partition[T]) -> float:
        """Measure the quality of the given partition P of the graph G, as defined by the Modularity quality function."""
        m = P.graph_size

        # For empty graphs (without edges) return NaN, as Modularity is not defined then, due to the division by `2*m`.)
        if m == 0:
            return float("NaN")

        norm: float = self.gamma / (2 * m)

        def community_summand(C: Set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the total weight of edges within that community:
            e_c: int = nx.induced_subgraph(P.G, C).size(weight=P._weight)
            # Also determine the total sum of node degrees in the community C
            deg_c: int = P.degree_sum(next(iter(C)))

            # From this, calculate the contribution of community c:
            # The "From Louvain to Leiden" paper doesn't state this, but for the modularity to match the original, cited definition, e_c
            # needs to be counted *twice*, as in an undirected graph, every edge {u,v} is counted twice, as (u,v) and as (v,u).
            return 2 * e_c - norm * deg_c**2

        # Calculate the constant potts model by adding the summands for all communities:
        return sum(map(community_summand, P)) / float(2 * m)

    def delta(self, P: Partition[T], v: T, target: Set[T]) -> float:
        """Measure the increase (or decrease, if negative) of this quality function when moving node v into the target community."""
        if v in target:
            return 0.0

        # First, determine the graph size
        m: int = P.graph_size
        # Now, calculate the difference in the source and target communities in the `E(C,C)` value for removing / adding v.
        source_community = P.node_community(v)
        diff_source = single_node_neighbor_cut_size(P.G, v, set(u for u in source_community if u != v), P._weight)
        diff_target = single_node_neighbor_cut_size(P.G, v, target, P._weight)

        # Get the necessary degrees
        deg_v: int = P.G.degree(v, weight=P._weight)
        degs_source: int = P.degree_sum(v)
        degs_target: int = P.degree_sum(next(iter(target))) if target else 0

        # Now, calculate and return the difference of the metric that will be accrued by moving the node v into the community t.
        # For the derivation see the appendix of the accompanying project documentation, here it is slightly rearranged.
        # Note that we divide by m instead of 2*m here, as we want the delta function compatible to the calculation in __call__,
        # which in turn is implemented to be compatible with NetworkX, as described above.
        return ((diff_target - diff_source) - self.gamma / (2 * m) * (deg_v**2 + deg_v * (degs_target - degs_source))) / m


class CPM(QualityFunction[T], Generic[T]):
    """Implementation of the Constant Potts Model (CPM) as a quality function."""

    def __init__(self, gamma: float = 0.25) -> None:
        """Create a new instance of the Constant Potts Model with the given resolution parameter gamma."""
        self.gamma: float = gamma

    def __call__(self, P: Partition[T]) -> float:
        """Measure the quality of the given partition P of the graph G, as defined by the CPM quality function."""

        def community_summand(C: Set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the total weight of edges within that community:
            e_c: int = nx.induced_subgraph(P.G, C).size(weight=P._weight)
            # Also get the number of nodes in this community.
            node_weights = P.G.nodes.data(P._weight, default=1)
            n_c: int = sum(node_weights[u] for u in C)
            pairs: int = comb(n_c, 2)

            # From this, calculate the contribution of community c:
            return e_c - self.gamma * pairs

        # Calculate the constant potts model by adding the summands for all communities:
        return sum(map(community_summand, P))

    def delta(self, P: Partition[T], v: T, target: Set[T]) -> float:
        """Measure the increase (or decrease, if negative) of this quality function when moving node v into the target community."""
        if v in target:
            return 0.0

        # First calculate the difference in the source and target communities in the `E(C,C)` value for removing / adding v.
        source_community = P.node_community(v)
        diff_source = single_node_neighbor_cut_size(P.G, v, set(u for u in source_community if u != v), P._weight)
        diff_target = single_node_neighbor_cut_size(P.G, v, target, P._weight)

        # Determine the weight of v and the total weights of the source community (with v) and the target community (without v)
        node_weights = P.G.nodes.data(P._weight, default=1)
        v_weight: float = node_weights[v]
        source_weight: float = sum(node_weights[u] for u in source_community)
        target_weight: float = sum(node_weights[u] for u in target)

        # Now, calculate and return the difference of the metric that will be accrued by moving the node v into the community t:
        return diff_target - diff_source - self.gamma * v_weight * (v_weight + target_weight - source_weight)
