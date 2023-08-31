"""This module defines quality metrics and provides implementations of Modularity and the ConstantPotts Model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import networkx as nx
from networkx import Graph

from .utils import Partition

T = TypeVar("T")


class QualityMetric(ABC, Generic[T]):
    """A metric that, when called, measures the quality of a partition into communities."""

    @abstractmethod
    def __call__(self, G: Graph, 𝓟: Partition[T]) -> float:
        """Measure the quality of the given partition as applied to the graph provided."""
        raise NotImplementedError()

    def delta(self, G: Graph, 𝓟: Partition[T], v: T, target: set[T] | frozenset[T], baseline: None | float = None) -> float:
        """Measure the increase (or decrease, if negative) of this quality metric when moving node v into the target community."""
        if not baseline:
            baseline = self(G, 𝓟)
        moved = 𝓟.move_node(v, target)
        return self(G, moved) - baseline


class Modularity(QualityMetric[T], Generic[T]):
    """Implementation of Modularity as a quality function."""

    def __init__(self, γ: float = 0.25):
        """Create a new instance of Modularity quality metric with the given resolution parameter γ."""
        self.γ = γ

    def __call__(self, G: Graph, 𝓟: Partition[T]) -> float:
        """Measure the quality of the given partition 𝓟 of the graph G, as defined by the Modularity quality metric."""
        node_degrees = dict(G.degree(weight=None))
        two_m = sum(node_degrees.values())

        # For empty graphs (without edges) return NaN, as Modularity is not defined then, due to the division by `2*m`.)
        if two_m == 0:
            return float('NaN')

        norm: float = self.γ / two_m

        def community_summand(C: set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the number of edges within that community:
            e_c: int = len(nx.induced_subgraph(G, C).edges)
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
        self.γ = γ

    def __call__(self, G: Graph, 𝓟: Partition[T]) -> float:
        """Measure the quality of the given partition 𝓟 of the graph G, as defined by the CPM quality metric."""

        def community_summand(C: set[T]) -> float:
            # Calculate the summand representing the community `c`.
            # First, determine the number of edges within that community:
            e_c: int = len(nx.induced_subgraph(G, C).edges)
            # Also get the number of nodes in this community.
            n_c: int = len(C)

            # From this, calculate the contribution of community c:
            return e_c - self.γ * (n_c * (n_c - 1) / 2)

        # Calculate the constant potts model by adding the summands for all communities:
        return sum(map(community_summand, 𝓟))
