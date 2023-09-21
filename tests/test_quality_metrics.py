from math import isnan

import networkx as nx

from community_detection.quality_functions import CPM, Modularity, QualityFunction
from community_detection.utils import Partition, freeze

from .utils import partition_randomly

PRECISION = 2e-15

# Don't let black destroy the manual formatting in this document:
# fmt: off

def test_modularity_trivial_values() -> None:
    """Test modularity calculation for special graphs and partitions to see if the values match our expectation."""
    C = nx.complete_graph(10)
    P = Partition.from_partition(C, [{i for i in range(10)}])
    Q = Partition.from_partition(C, [{i} for i in range(10)])

    H: QualityFunction[int] = Modularity(1)

    assert 0.0 == H(P)
    assert abs(-0.1 - H(Q)) < PRECISION

    # For empty graphs, the modularity is not defined. We return NaN in this case:
    E = nx.empty_graph(10)
    P = Partition.from_partition(E, P)
    assert isnan(H(P))
    Q = Partition.from_partition(E, Q)
    assert isnan(H(Q))


def test_modularity_example() -> None:
    """Test the Modularity calculation with a few examples."""
    # Produce the example graph in the wikipedia page on modularity
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 4), (0, 7),
        (1, 2), (2, 3), (3, 1),
        (4, 5), (5, 6), (6, 4),
        (7, 8), (8, 9), (9, 7)
    ])

    H: QualityFunction[int] = Modularity(1)

    # Start with the (original) singleton partition
    P = Partition.from_partition(G, [{0, 1, 2, 3}, {4, 5, 6}, {7, 8, 9}])

    print(f"{H(P)=}")
    expected = 0.4896
    assert abs(H(P) - expected) < 1E-4, f"{H(P)=} != {expected}=expected"


def test_modularity_random() -> None:
    """Test the Modularity calculation with random partitions of the karate club graph."""
    G = nx.karate_club_graph()
    nodes = list(G.nodes)

    H: QualityFunction[int] = Modularity(1)

    for _ in range(10):
        # Start with the (original) singleton partition
        P = Partition.from_partition(G, partition_randomly(nodes))

        our_mod = H(P)
        reference_mod = nx.community.modularity(G, P.as_set(), weight=None, resolution=1)

        assert abs(our_mod - reference_mod) < PRECISION, f"H(P) = {our_mod} != {reference_mod} = expected"


def test_modularity_delta() -> None:
    """Test the Modularity.delta() calculation."""
    # Produce the weighted (4,0)-barbell graph described in the supplementary information of "louvain to leiden", p. 6
    B = nx.Graph()
    B.add_weighted_edges_from([
        (0, 1, 3),
        (0, 2, 1.5), (0, 3, 1.5), (0, 4, 1.5), (2, 3, 3), (2, 4, 3), (3, 4, 3),
        (1, 5, 1.5), (1, 6, 1.5), (1, 7, 1.5), (5, 6, 3), (5, 7, 3), (6, 7, 3)
    ])

    H: QualityFunction[int] = Modularity(0.95)

    # Start with the (original) singleton partition
    P = Partition.from_partition(B, [{0, 1, 6}, {2, 3, 4}, {5, 7}], weight="weight")

    # Initialize the variable in which we will accumulate the delta values
    old_value = H(P)

    # A sequence of move sequences, described as tuples of a node and the community to move it into
    # The first move moves a node into its current community (i.e. a no-op) - we expect a delta of 0 to be calculated here
    moves: list[tuple[int, set[int]]] = [
        (0, {0, 1, 6}), (1, {5, 7}), (0, set()), (6, {1, 5, 7}), (2, {0}), (3, {0, 2}), (4, {0, 2, 3}), (0, {1, 5, 6, 7}), (1, set()), (0, {1})
    ]

    # Now, carry out the moves and compare the projected and actual differences for each move
    for move in moves:
        delta = H.delta(P, move[0], move[1])
        P.move_node(*move)

        new_value = H(P)
        assert abs((new_value - old_value) - delta) < PRECISION, \
            f"Projected Modularity-delta {delta} did not match actual delta {(new_value - old_value)} in move {move}!"
        old_value = new_value

    # Sanity check that our node movements produced the expected state
    assert P.as_set() == freeze([{0, 1}, {2, 3, 4}, {5, 6, 7}])


def test_cpm_trivial_values() -> None:
    """Test CPM calculation for some trivial  graphs and partitions to see if the values match the expectation."""
    C = nx.complete_graph(10)
    E = nx.empty_graph(10)
    P_C = Partition.from_partition(C, [{i for i in range(10)}])
    P_E = Partition.from_partition(E, [{i for i in range(10)}])
    Q_C = Partition.from_partition(C, [{i} for i in range(10)])
    Q_E = Partition.from_partition(E, [{i} for i in range(10)])

    H: QualityFunction[int] = CPM(0.25)

    # Values calculated manually for gamma = 0.25:
    assert -11.25 == H(P_E)  # The empty graph (no edges) with the trivial partition has CPM -11.25
    assert   0.00 == H(Q_E)  # Empty graph with singleton partition has CPM 0 (better than the trivial partition)
    assert   0.00 == H(Q_C)  # Complete graph K_10 with singleton partition has CPM 0
    assert  33.75 == H(P_C)  # The graph K_10 with the trivial partition has CPM 33.75 (improves singleton partition)


def test_cpm_example_from_material() -> None:
    """Compare the calculation of the CPM metric with known-good values from the source material."""
    # Produce the weighted (4,0)-barbell graph described in the supplementary information of "louvain to leiden", p. 6
    B = nx.Graph()
    B.add_weighted_edges_from([
        (0, 1, 3),
        (0, 2, 1.5), (0, 3, 1.5), (0, 4, 1.5), (2, 3, 3), (2, 4, 3), (3, 4, 3),
        (1, 5, 1.5), (1, 6, 1.5), (1, 7, 1.5), (5, 6, 3), (5, 7, 3), (6, 7, 3)
    ])

    # Produce partitions with and without weight information
    M = Partition.from_partition(B, [{0, 2, 3, 4}, {1, 5, 6, 7}])
    N = Partition.from_partition(B, [{2, 3, 4}, {0, 1}, {5, 6, 7}])
    M_w = Partition.from_partition(B, [{0, 2, 3, 4}, {1, 5, 6, 7}], "weight")
    N_w = Partition.from_partition(B, [{2, 3, 4}, {0, 1}, {5, 6, 7}], "weight")

    H: QualityFunction[int] = CPM(1.0)

    # Values calculated manually for and the (4,0)-barbell graph:
    # Unweighted (does not correspond to supplementary information)
    assert H(M) == 0
    assert H(N) == 0
    # Weighted (as in the supplementary material)
    assert H(M_w) == 15
    assert H(N_w) == 14


def test_cpm_delta() -> None:
    """Test the CPM.delta() calculation by transforming one partition into another."""
    # Produce the weighted (4,0)-barbell graph described in the supplementary information of "louvain to leiden", p. 6
    B = nx.Graph()
    B.add_weighted_edges_from([
        (0, 1, 3),
        (0, 2, 1.5), (0, 3, 1.5), (0, 4, 1.5), (2, 3, 3), (2, 4, 3), (3, 4, 3),
        (1, 5, 1.5), (1, 6, 1.5), (1, 7, 1.5), (5, 6, 3), (5, 7, 3), (6, 7, 3)
    ])

    H: QualityFunction[int] = CPM(0.95)

    # Start with the (original) singleton partition
    P = Partition.from_partition(B, [{0, 1, 6}, {2, 3, 4}, {5, 7}], "weight")

    # Initialize the variable in which we will accumulate the delta values
    old_value = H(P)

    # A sequence of move sequences, described as tuples of a node and the community to move it into
    # The first move moves a node into its current community (i.e. a no-op) - we expect a delta of 0 to be calculated here
    moves: list[tuple[int, set[int]]] = [
        (0, {0, 1, 6}), (1, {5, 7}), (0, set()), (6, {1, 5, 7}), (2, {0}), (3, {0, 2}), (4, {0, 2, 3}), (0, {1, 5, 6, 7}), (1, set()), (0, {1})
    ]

    # Now, carry out the moves and compare the projected and actual differences for each move
    for move in moves:
        delta = H.delta(P, move[0], move[1])
        P.move_node(*move)

        new_value = H(P)
        assert abs((new_value - old_value) - delta) < PRECISION, \
            f"Projected CPM-delta {delta} did not match actual delta {(new_value - old_value)} in move {move}!"
        old_value = new_value

    # Sanity check that our node movements produced the expected state
    assert P.as_set() == freeze([{0, 1}, {2, 3, 4}, {5, 6, 7}])
