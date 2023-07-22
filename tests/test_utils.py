import networkx as nx
import pytest

from ..leiden import leiden
from ..louvain import louvain
from ..utils import Partition, freeze, recursive_size, flat, flatₚ, argmax, aggregate_graph, singleton_partition


def test_partition_creation():
    E = nx.generators.empty_graph(0)
    G = nx.generators.classic.complete_graph(5)
    H = nx.generators.barbell_graph(5, 2)

    # Check that we can create valid partitions for the graphs above
    assert Partition(E, [{}]) is not None
    assert Partition(G, [{0, 1, 2, 3, 4}]) is not None
    assert Partition(G, [{0}, {1}, {2}, {3}, {4}]) is not None
    assert Partition(H, [{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}]) is not None

    # Now check that partition creation fails when given sets which don't form a partition of the graph's nodes:
    # Partition contains nodes not in the graph:
    with pytest.raises(AssertionError):
        Partition(E, [{0}])
    # Not all graph nodes are present:
    with pytest.raises(AssertionError):
        Partition(G, [{0, 1, 3, 4}])  # Missing 2
    # There is a non-empty intersection of two sets in the partition
    with pytest.raises(AssertionError):
        Partition(G, [{0, 1, 2}, {2, 3, 4}])


def test_partition_moving():
    G = nx.generators.classic.complete_graph(5)
    P = [{0, 1, 2, 3}, {4}]

    𝓟 = Partition(G, P)              # Start with the partition indicated in P and do each of the following:
    𝓠 = 𝓟.move_node(0, {})           # a) Move node 0 to its own community (i.e. nothing should change)
    𝓡 = 𝓟.move_node(0, {4})          # b) Move node 0 to the community which contains node 4
    𝓢 = 𝓟.move_node(4, {0, 1, 2, 3}) # c) Move node 0 to the community containing all other nodes

    # Now, verify that both the communities and the membership of node 4 are correct:
    assert 𝓟.node_community(4) == {4}
    assert 𝓟.as_set() == freeze([{0, 1, 2, 3}, {4}])

    assert 𝓠.node_community(4) == {4}
    assert 𝓠.as_set() == freeze([{1, 2, 3}, {4}, {0}])

    assert 𝓡.node_community(4) == {0, 4}
    assert 𝓡.as_set() == freeze([{1, 2, 3}, {0, 4}])

    assert 𝓢.node_community(4) == {0, 1, 2, 3, 4}
    assert 𝓢.as_set() == freeze([{0, 1, 2, 3, 4}])


def test_freeze():
    pass


def test_recursive_size():
    assert recursive_size([]) == 0

    assert recursive_size(42) == 1
    assert recursive_size([42]) == 1

    assert recursive_size([0, [1]]) == 2
    assert recursive_size([[[[0, [1]]]]]) == 2

    assert recursive_size([[], 1, [2], [[3]]]) == 3
    assert recursive_size([1, 2, 3]) == 3


def test_flat():
    # Note that '{}' is not an empty set, but an empty dict!
    assert flat(set()) == set()  # test the input {}
    assert flat({ 0, 1, 2, 3 }) == {0, 1, 2, 3}  # test the input {0, 1, 2, 3}

    assert flat({ frozenset( set() ) }) == set()  # test the input { set() }
    assert flat({ frozenset( frozenset( set() ) ) }) == set()  # test the input { { { } } }
    assert flat({ 0, frozenset( {1} ), frozenset( {2, frozenset({3}) } ) }) == {0, 1, 2, 3}  # test the input { 0, {1}, {2, {3}} }


def test_flat_partition():
    pass


def test_argmax():
    pass


def test_aggregate_graph():
    pass


def test_singleton_partition():
    pass
