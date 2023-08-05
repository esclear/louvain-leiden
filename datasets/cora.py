import os

import networkx as nx
import pandas as pd

def get_graph() -> nx.Graph:
    data_path = os.path.join(os.path.dirname(__file__), 'cora')
    nodes = pd.read_csv(os.path.join(data_path, "cora.content"), sep='\t', header=None, usecols=[0, 1434], names=["node", "subject"])
    edges = pd.read_csv(os.path.join(data_path, "cora.cites"), sep='\t', header=None, names=["target", "source"])

    G = nx.from_pandas_edgelist(edges)
    subject_map = {l["node"]: l["subject"] for i, l in nodes.iterrows()}
    nx.set_node_attributes(G, subject_map, "subject")

    return G
