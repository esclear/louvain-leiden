import os

import networkx as nx

def get_graph() -> nx.Graph:
    path = os.path.join(os.path.dirname(__file__), 'jazz/out.arenas-jazz')
    G = nx.read_edgelist(path, comments='%', create_using=nx.Graph(), nodetype=int)

    return G
