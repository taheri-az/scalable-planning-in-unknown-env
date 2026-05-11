import numpy as np
import networkx as nx
from collections import defaultdict


def generate_product_automaton(nodes, edges, adj_matrix, dfa_states, dfa_transitions, observations):
    physical_nodes = list(nodes.flatten())

    neighbors = {}
    for node in physical_nodes:
        neighbors[node] = [nbr for nbr in physical_nodes if adj_matrix[node][nbr] == 1]

    dfa_map = {s: [] for s in dfa_states}
    for (src, conds, tgt) in dfa_transitions:
        dfa_map[src].append((conds, tgt))

    product_graph = nx.DiGraph()
    product_nodes = []
    product_idx = {}

    for p_node in physical_nodes:
        for q_state in dfa_states:
            st = (int(p_node), q_state)
            product_idx[st] = len(product_nodes)
            product_nodes.append(st)
            product_graph.add_node(st)

    N = len(product_nodes)
    product_adj_matrix = np.zeros((N, N), dtype=np.int8)

    transitions = []

    for p_node in physical_nodes:
        for q_state in dfa_states:
            src_state = (int(p_node), q_state)

            for (conds, q_next) in dfa_map[q_state]:
                if not any(obs in observations for obs in conds):
                    continue

                for p_next in neighbors[p_node]:
                    tgt_state = (int(p_next), q_next)

                    transitions.append((src_state, tgt_state))
                    product_graph.add_edge(src_state, tgt_state)

                    i = product_idx[src_state]
                    j = product_idx[tgt_state]
                    product_adj_matrix[i][j] = 1

    return product_graph, transitions, product_nodes, product_adj_matrix
