"""This module defines quality metrics and provides implementations of Modularity and the ConstantPotts Model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from math import comb
from typing import Generic, TypeVar

import networkx as nx
from networkx import Graph

from .utils import Partition

T = TypeVar("T")


class QualityMetric(ABC, Generic[T]):
    """A metric that, when called, measures the quality of a partition into communities."""

    @abstractmethod
    def __call__(self, G: Graph, 𝓟: Partition[T], weight: None | str = None) -> float:
        """Measure the quality of the given partition as applied to the graph provided."""
        raise NotImplementedError()

    def delta(
        self, G: Graph, 𝓟: Partition[T], v: T, target: set[T] | frozenset[T], weight: None | str = None, baseline: None | float = None
    ) -> float:
        """Measure the increase (or decrease, if negative) of this quality metric when moving node v into the target community."""
        if not baseline:
            baseline = self(G, 𝓟, weight)
        moved = copy(𝓟).move_node(v, target)
        return self(G, moved, weight) - baseline


class Modularity(QualityMetric[T], Generic[T]):
    """Implementation of Modularity as a quality function."""

    def __init__(self, γ: float = 0.25):
        """Create a new instance of Modularity quality metric with the given resolution parameter γ."""
        self.γ: float = γ

    def __call__(self, G: Graph, 𝓟: Partition[T], weight: None | str = None) -> float:
        """Measure the quality of the given partition 𝓟 of the graph G, as defined by the Modularity quality metric."""
        node_degrees = dict(G.degree(weight=None))
        two_m = 2 * G.size()

        # For empty graphs (without edges) return NaN, as Modularity is not defined then, due to the division by `2*m`.)
        if two_m == 0:
            return float('NaN')

        norm: float = self.γ / two_m

        def community_summand(C: set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the number of edges within that community:
            e_c: int = nx.induced_subgraph(G, C).size()
            # Sum up the degrees of nodes in the community
            degree_sum: int = sum(node_degrees[u] for u in C)

            # From this, calculate the contribution of community c:
            return 2 * e_c - norm * degree_sum**2

        # Calculate the modularity by adding the summands for all communities and dividing by `2 * m`:
        return sum(map(community_summand, 𝓟)) / two_m


class CPM(QualityMetric[T], Generic[T]):
    """Implementation of the Constant Potts Model (CPM) as a quality function."""

    def __init__(self, γ: float = 0.25):
        """Create a new instance of the Constant Potts Model with the given resolution parameter γ."""
        self.γ: float = γ

    def __call__(self, G: Graph, 𝓟: Partition[T], weight: None | str = None) -> float:
        """Measure the quality of the given partition 𝓟 of the graph G, as defined by the CPM quality metric."""

        def community_summand(C: set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the number of edges within that community:
            e_c: int = nx.induced_subgraph(G, C).size(weight=weight)
            # Also get the number of nodes in this community.
            node_weights = G.nodes.data(weight, default=1)
            n_c: int = sum(node_weights[u] for u in C) # TODO Check that this is used

            # From this, calculate the contribution of community c:
            return e_c - self.γ * comb(n_c, 2)

        # Calculate the constant potts model by adding the summands for all communities:
        return sum(map(community_summand, 𝓟))

    def delta(self, G: Graph, 𝓟: Partition[T], v: T, target: set[T] | frozenset[T], weight: None | str = None, baseline: None | float = None) -> float:
        """Measure the increase (or decrease, if negative) of this quality metric when moving node v into the target community."""
        # First calculate the difference in the source and target communities in the `E(C,C)` value for removing / adding v.
        source_community = 𝓟.node_community(v)
        diff_source = nx.cut_size(G, [v], source_community - {v}, weight)
        diff_target = nx.cut_size(G, [v], target, weight)

        # Determine the weight of v and the total weights of the source community (with v) and the target community (without v)
        node_weights = G.nodes.data(weight, default=1)
        v_weight = node_weights[v]
        source_weight = sum(node_weights[u] for u in source_community)
        target_weight = sum(node_weights[u] for u in target)

        # Now, calculate and return the difference of the metric that will be accrued by moving the node v into the community t:
        return diff_target - diff_source + self.γ * \
            (comb(source_weight, 2) + comb(target_weight, 2) - comb(source_weight - v_weight, 2) - comb(target_weight + v_weight, 2))