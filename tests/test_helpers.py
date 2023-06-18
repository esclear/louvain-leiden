import networkx as nx

from ..leiden import leiden
from ..louvain import louvain
from ..utils import *


def test_partition_creation():
    pass


def test_partition_moving():
    G = nx.generators.classic.complete_graph(5)
    P = [{0, 1, 2, 3}, {4}]

    𝓟 = Partition(G, P)
    𝓠 = 𝓟.move_node(0, {})
    𝓡 = 𝓟.move_node(0, {4})
    𝓢 = 𝓟.move_node(4, {0, 1, 2, 3})

    assert 𝓟.sets == [{0, 1, 2, 3}, {4}]
    assert 𝓠.sets == [{1, 2, 3}, {4}, {0}]
    assert 𝓡.sets == [{1, 2, 3}, {0, 4}]
    assert 𝓢.sets == [{0, 1, 2, 3, 4}]


def test_flat():
    pass


def test_flat_partition():
    pass
