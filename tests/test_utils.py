import networkx as nx
import pytest

from community_detection.leiden import leiden
from community_detection.louvain import louvain
from community_detection.utils import Partition, aggregate_graph, argmax, flat, flatₚ, freeze, recursive_size, singleton_partition

# Don't let black destroy the manual formatting in this document:
# fmt: off

def test_partition_creation():
    E = nx.generators.empty_graph(0)
    G = nx.generators.classic.complete_graph(5)
    H = nx.generators.barbell_graph(5, 2)

    # Check that we can create valid partitions for the graphs above
    𝓟: Partition[int] = Partition(E, [])
    assert 𝓟 is not None
    assert 𝓟.communities == ()

    𝓠 = Partition(G, [{0, 1, 2, 3, 4}])
    assert 𝓠 is not None
    assert 𝓠.communities == ({0, 1, 2, 3, 4},)

    𝓡 = Partition(G, [{0}, {1}, {2}, {3}, {4}])
    assert 𝓡 is not None
    assert 𝓡.communities == ({0}, {1}, {2}, {3}, {4})

    𝓢 = Partition(H, [{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}])
    assert 𝓢 is not None
    assert 𝓢.communities == ({0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11})

    assert len(𝓟) == 0
    assert len(𝓠) == 1
    assert len(𝓡) == 5
    assert len(𝓢) == 3

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
    𝓠 = 𝓟.move_node(0, set())        # a) Move node 0 to its own community (i.e. nothing should change)
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
    assert freeze([]) == set()
    assert freeze([set()]) == { frozenset(set()) }
    assert freeze([{1, 2, 3}, {4, 5}, set()]) == { frozenset({1, 2, 3}), frozenset({4, 5}), frozenset() }
    assert freeze([set(), set()]) == { frozenset(set()) }


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
    # flatₚ is called on aggregate graphs, where every node of the aggregate graph represents (potentially arbitrarily nested) sets
    # of nodes in the original graph.

    # First, check with a simple graph
    G = nx.generators.classic.complete_graph(10)

    𝓟 = singleton_partition(G)  # singleton partition
    𝓠 = Partition(G, [{ *G.nodes }])  # trivial partition (all nodes in one community)

    # To compare properly, we use the freeze function, so that we can compare sets, where the order doesn't matter.
    assert freeze(flatₚ(𝓟)) == freeze([{i} for i in range(10)])
    assert freeze(flatₚ(𝓠)) == freeze([{i for i in range(10)}])

    # Calculate an aggregate graph by repeatedly merging:
    𝓡 = Partition(G, [ {0, 1, 2}, {3, 4}, {5, 6}, {7, 8}, {9} ])
    H = aggregate_graph(G, 𝓡)

    𝓢 = Partition(H, [
        { frozenset({0, 1, 2}), frozenset({3, 4}) },
        { frozenset({5, 6}) },
        { frozenset({7, 8}), frozenset({9}) }
    ])

    𝓣 = Partition(H, S)

    assert freeze(flatₚ(𝓣)) == freeze([ {0, 1, 2, 3, 4} , {5, 6}, {7, 8, 9}])


def test_argmax():
    assert argmax(lambda x: x, None) is None
    assert argmax(lambda x: x, []) is None
    assert argmax(lambda x: x, set()) is None

    # argmax returns tuples of the form (arg, value, index)
    # check that for constant arguments and values the first index (0) is chosen:
    assert argmax(lambda x: 42, [10 for i in range(10)]) == (10, 42, 0)
    # check that for variable arguments but constant values the first index (0) is chosen:
    assert argmax(lambda x: 42, [10 + i for i in range(10)]) == (10, 42, 0)
    # check that the calculations are carried out properly
    #   -> at indices 0..9 we have the inputs 10..19 and the values 30..39
    assert argmax(lambda x: 20 + x, [10 + i for i in range(10)]) == (19, 39, 9)
    #   -> at indices 0..9 we have the inputs 10..19 and the values 40..31
    assert argmax(lambda x: 30 - x, [10 + i for i in range(10)]) == (10, 20, 0)
    # check that finding a minimum in the middle of the list works as well and the first occurrence is returned
    assert argmax(lambda x: x, [0, 1, 3, 8, 5, 8, 6]) == (8, 8, 3)


def test_aggregate_graph():
    G = nx.generators.classic.complete_graph(5)
    communities = [{0}, {1, 2}, {3, 4}]
    𝓟 = Partition(G, communities)

    H = aggregate_graph(G, 𝓟)

    # Short sanity check: We have three nodes, representing the three communities
    # and as many edges as before (recall that the aggregate graph H is a multigraph!)
    assert len(H.nodes()) == 3
    assert len(H.edges()) == 10

    # Verify that the nodes of the aggregate graph correspond to the communities
    assert set(H.nodes()) == freeze(communities)
    # Check that the inter-community-edges are correct
    assert H.number_of_edges(frozenset({0}),    frozenset({1, 2})) == 2
    assert H.number_of_edges(frozenset({0}),    frozenset({3, 4})) == 2
    assert H.number_of_edges(frozenset({1, 2}), frozenset({3, 4})) == 4
    # Also check that self-loops for the communities are correct
    assert H.number_of_edges(frozenset({1, 2}), frozenset({1, 2})) == 1
    assert H.number_of_edges(frozenset({3, 4}), frozenset({3, 4})) == 1


def test_singleton_partition():
    E = nx.generators.empty_graph(0)
    G = nx.generators.classic.complete_graph(5)
    H = nx.generators.barbell_graph(5, 2)

    𝓟 = singleton_partition(E)
    𝓠 = singleton_partition(G)
    𝓡 = singleton_partition(H)

    assert 𝓟.as_set() == freeze([])
    assert 𝓠.as_set() == freeze([{0}, {1}, {2}, {3}, {4}])
    assert 𝓡.as_set() == freeze([{i} for i in range(12)])
