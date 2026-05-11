import numpy as np
import re


def create_grid(n, m):
    grid = np.zeros((n, m), dtype=int)
    return grid


def create_graph(n, m):
    num_nodes = n * m
    nodes = np.arange(num_nodes).reshape(n, m)
    edges = []

    for i in range(n):
        for j in range(m - 1):
            edges.append((nodes[i][j], nodes[i][j + 1]))
            edges.append((nodes[i][j + 1], nodes[i][j]))

    for j in range(m):
        for i in range(n - 1):
            edges.append((nodes[i][j], nodes[i + 1][j]))
            edges.append((nodes[i + 1][j], nodes[i][j]))

    for i in range(n):
        for j in range(m):
            edges.append((nodes[i][j], nodes[i][j]))

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    return nodes, edges, adj_matrix


def grid_probabilities(grid, n, m):
    result = []
    for i in range(n * m):
        cel_prob = [grid[i][j][0] for j in range(len(grid[i]))]
        result.append(cel_prob)
    return result


def getting_physical(optimal_traj):
    physical_seq = []
    for state in optimal_traj:
        physical = int(state.split(",")[0].strip("()"))
        physical_seq.append(physical)
    return physical_seq
