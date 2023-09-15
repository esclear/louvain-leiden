"""This module defines quality metrics and provides implementations of Modularity and the ConstantPotts Model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from math import comb
from typing import Generic, TypeVar

import networkx as nx
from networkx import Graph

from .utils import Partition, single_node_neighbor_cut_size

T = TypeVar("T")


class QualityMetric(ABC, Generic[T]):
    """A metric that, when called, measures the quality of a partition into communities."""

    @abstractmethod
    def __call__(self, 𝓟: Partition[T]) -> float:
        """Measure the quality of the given partition as applied to the graph provided."""
        raise NotImplementedError()

    def delta(self, 𝓟: Partition[T], v: T, target: set[T] | frozenset[T]) -> float:
        """Measure the increase (or decrease, if negative) of this quality metric when moving node v into the target community."""
        baseline = self(𝓟, 𝓟.weight)
        moved = copy(𝓟).move_node(v, target)
        return self(G, moved, 𝓟.weight) - baseline


class Modularity(QualityMetric[T], Generic[T]):
    """Implementation of Modularity as a quality function."""

    def __init__(self, γ: float = 0.25):
        """Create a new instance of Modularity quality metric with the given resolution parameter γ."""
        self.γ: float = γ

    def __call__(self, 𝓟: Partition[T]) -> float:
        """Measure the quality of the given partition 𝓟 of the graph G, as defined by the Modularity quality metric."""
        m = 𝓟.graph_size

        # For empty graphs (without edges) return NaN, as Modularity is not defined then, due to the division by `2*m`.)
        if m == 0:
            return float('NaN')

        norm: float = self.γ / (2 * m)

        def community_summand(C: set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the total weight of edges within that community:
            e_c = nx.induced_subgraph(𝓟.G, C).size(weight=𝓟._weight)  # TODO: Can this be cached
            # Also determine the total sum of node degrees in the community C
            deg_c = 𝓟.degree_sum(next(iter(C)))

            # From this, calculate the contribution of community c:
            # The "From Louvain to Leiden" paper doesn't state this, but for the modularity to match the original, cited definition, e_c
            # needs to be counted *twice*, as in an undirected graph, every edge {u,v} is counted twice, as (u,v) and as (v,u).
            return 2 * e_c - norm * deg_c**2

        # Calculate the constant potts model by adding the summands for all communities:
        return sum(map(community_summand, 𝓟)) / (2 * m)

    def delta(self, 𝓟: Partition[T], v: T, target: set[T] | frozenset[T]) -> float:
        """Measure the increase (or decrease, if negative) of this quality metric when moving node v into the target community."""
        if v in target:
            return 0.0

        # First, determine the graph size
        m = 𝓟.graph_size
        # Now, calculate the difference in the source and target communities in the `E(C,C)` value for removing / adding v.
        source_community = 𝓟.node_community(v)
        diff_source = single_node_neighbor_cut_size(𝓟.G, v, set(u for u in source_community if u != v), 𝓟._weight)
        diff_target = single_node_neighbor_cut_size(𝓟.G, v, target, 𝓟._weight)

        # Get the necessary degrees
        deg_v = 𝓟.G.degree(v, weight=𝓟._weight)
        degs_source = 𝓟.degree_sum(v)
        degs_target = 𝓟.degree_sum(next(iter(target))) if target else 0

        # Now, calculate and return the difference of the metric that will be accrued by moving the node v into the community t:
        return (diff_target - diff_source + self.γ / (2 * m) * (deg_v * (degs_source - degs_target) - deg_v**2)) / m


class CPM(QualityMetric[T], Generic[T]):
    """Implementation of the Constant Potts Model (CPM) as a quality function."""

    def __init__(self, γ: float = 0.25):
        """Create a new instance of the Constant Potts Model with the given resolution parameter γ."""
        self.γ: float = γ

    def __call__(self, 𝓟: Partition[T]) -> float:
        """Measure the quality of the given partition 𝓟 of the graph G, as defined by the CPM quality metric."""

        def community_summand(C: set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the total weight of edges within that community:
            e_c = nx.induced_subgraph(𝓟.G, C).size(weight=𝓟._weight)
            # Also get the number of nodes in this community.
            node_weights = 𝓟.G.nodes.data(𝓟._weight, default=1)
            n_c: int = sum(node_weights[u] for u in C)

            # From this, calculate the contribution of community c:
            return e_c - self.γ * comb(n_c, 2)

        # Calculate the constant potts model by adding the summands for all communities:
        return sum(map(community_summand, 𝓟))

    def delta(self, 𝓟: Partition[T], v: T, target: set[T] | frozenset[T]) -> float:
        """Measure the increase (or decrease, if negative) of this quality metric when moving node v into the target community."""
        if v in target:
            return 0.0

        # First calculate the difference in the source and target communities in the `E(C,C)` value for removing / adding v.
        source_community = 𝓟.node_community(v)
        diff_source = single_node_neighbor_cut_size(𝓟.G, v, set(u for u in source_community if u != v), 𝓟._weight)
        diff_target = single_node_neighbor_cut_size(𝓟.G, v, target, 𝓟._weight)

        # Determine the weight of v and the total weights of the source community (with v) and the target community (without v)
        node_weights = 𝓟.G.nodes.data(𝓟._weight, default=1)
        v_weight = node_weights[v]
        source_weight = sum(node_weights[u] for u in source_community)
        target_weight = sum(node_weights[u] for u in target)

        # Now, calculate and return the difference of the metric that will be accrued by moving the node v into the community t:
        return diff_target - diff_source + self.γ * ( comb(source_weight, 2) + comb(target_weight, 2)
            - comb(source_weight - v_weight, 2) - comb(target_weight + v_weight, 2))  # fmt: skip
