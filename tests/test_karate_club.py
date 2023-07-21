import networkx as nx

from ..leiden import leiden
from ..louvain import louvain
from ..utils import *


def get_graph():
    """
    Generate a representation of the Karate Club graph we can work with.
    """
    G = nx.karate_club_graph()
    return G
