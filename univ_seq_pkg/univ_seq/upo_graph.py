#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# create fully connected undirected graph
import numpy as np
import torch
#from torch_geometric.data import HeteroData
from torch_geometric.data import Data
#from torch_geometric.utils import to_undirected
#from scipy.special import comb as nchoosek
from univ_seq.polar.upo import UPO
from univ_seq.polar.sequences import int2bin

def get_max_group_size(N):
    n = int(np.log2(N))
    k = n//2
    return int(nchoosek(n,k))

############################################################
# graph construction
############################################################
def gen_upo_graph(N_max, N, K, SNRdB, fbm, node_indices, actions):
# create fully connected graph among the node_indices
# use binary representation as check node feature

    data = Data()

#    print('actions', actions)

    data.SNRdB = torch.tensor(SNRdB,dtype=torch.float32)
    data.N = torch.tensor(N)
    data.K = torch.tensor(K)

    # start from all frozen and start assigning info bits
    fbm = torch.tensor(fbm)
    data.num_info_assigned = torch.sum(fbm == 1)
    data.num_info_unassigned = data.K - data.num_info_assigned

    # node feature and type
    n = int(np.log2(N_max))
    in_vec = np.arange(N)
    ni = torch.tensor(in_vec)
    nx = int2bin(in_vec, n).copy()
    nx = torch.tensor(nx, dtype=torch.float)
    nt = fbm
    nm = torch.zeros((N,), dtype=torch.bool)
    nm[actions] = True

    data.x = nx
    data.node_type = nt
    data.action_mask = nm
    data.node_indices = ni
    data.num_actions = actions.size

#    n2n = []
#    for i in node_indices:
#        for j in node_indices:
#            if i != j: # no self loops
#                n2n.append(torch.tensor([i,j]))
#    if n2n:
#        n2n = torch.stack(n2n)
#        n2n = torch.transpose(n2n,0,1)
#    else:
#        n2n = torch.empty((2, 0), dtype=torch.int)

    # fast method to create fully connected graph
    n2n = np.array(np.meshgrid(node_indices, node_indices)).T.reshape(-1,2)
    # remove self loops
    n2n = n2n[ n2n[:,0] != n2n[:,1] ]

    n2n = torch.tensor(n2n)
    n2n = torch.transpose(n2n,0,1)

    data.edge_index = n2n

    return data


import copy

def from_gccmp_graph(data, N, K, SNRdB, fbm, reset=False):
    """ make a copy and modify N, K, SNRdB and fbm """

    data = copy.deepcopy(data)

    if reset:
        assert data.N == N
        data.SNRdB = torch.tensor(SNRdB,dtype=torch.float32)
        data.K = torch.tensor(K)
    else:
        assert data.N == N
        assert data.K == K

    fbm = torch.tensor(fbm)
    data.num_info_assigned = torch.sum(fbm == 1)
    data.num_info_unassigned = data.K - data.num_info_assigned

    nt = fbm
    data.node_type = nt

    return data

if __name__ == "__main__":

    N = 4
    K = 2
    SNRdB = 0.0
    fbm = [0,0,0,1]

    data = gen_gccmp_graph(N, K, SNRdB, fbm)

    SNRdB = 1.0
    fbm = [0,1,0,1]

    print(data)
    print(data.x)
    print(data.node_type)
    print(data.edge_index)

    data = from_gccmp_graph(data, N, K, SNRdB, fbm)

    print(data)
    print(data.x)
    print(data.node_type)
    print(data.edge_index)

    exit()

    ################################################################################
    # visualize network
    ################################################################################

    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    graph = to_networkx(data, to_undirected=False)
    print(graph)

    node_type_colors = {
        "variable": "#4599C3",
        "check": "#ED8546",
    }

    node_colors = []
    v_nodes = []
    labels = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        color = node_type_colors[node_type]
        node_colors.append(color)
        if attrs["type"] == "variable":
            labels[node] = f"v{node}"
            v_nodes.append(node)
        elif attrs["type"] == "check":
            labels[node] = f"c{node-N}"

    edge_type_colors = {
        ("variable", "v2c", "check"): "#8B4D9E",
        ("check", "c2v", "variable"): "#DFB825",
        ("check", "c2c", "check"): "#70B349",
    }

    edge_colors = []
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        color = edge_type_colors[edge_type]
        graph.edges[from_node, to_node]["color"] = color
        edge_colors.append(color)

    #pos = nx.bipartite_layout(graph, v_nodes)
    pos = nx.spring_layout(graph, k=2)
    #pos = nx.kamada_kawai_layout(graph)
    #pos = nx.nx_pydot.graphviz_layout(graph, prog="neato")

    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=600,
    )

    plt.show()




