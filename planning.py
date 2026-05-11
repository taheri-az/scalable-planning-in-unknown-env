import re
import copy
import time
import random
import numpy as np
from collections import defaultdict
from collections import deque
from labeling import get_states_within_h_distance


def compare_manhattan_distance(n, m, state1, state2, action):
    row1, col1 = divmod(state1, m)
    row2, col2 = divmod(state2, m)

    initial_distance = abs(row1 - row2) + abs(col1 - col2)

    if action == "up" and row1 > 0:
        row1 -= 1
    elif action == "down" and row1 < n - 1:
        row1 += 1
    elif action == "left" and col1 > 0:
        col1 -= 1
    elif action == "right" and col1 < m - 1:
        col1 += 1
    elif action == "stay":
        pass

    new_distance = abs(row1 - row2) + abs(col1 - col2)

    if new_distance < initial_distance:
        return "decrease"
    elif new_distance > initial_distance:
        return "increase"
    else:
        return "same"


def build_transition_map(transitions):
    transition_map = defaultdict(list)
    for j in transitions:
        start_state = j[0]
        next_phy = j[1][0]
        transition_map[(start_state, next_phy)].append(j)
    return transition_map


def get_next_state(m, n, z, action, adj_matrix):
    valid_actions = get_valid_actions(m, n, z, adj_matrix)

    if action not in valid_actions:
        return None

    action_effects = {
        'left': -1,
        'right': 1,
        'up': -n,
        'down': n,
        'stay': 0,
    }

    delta = action_effects[action]
    next_state = z + delta
    return next_state


def calculate_manhattan_distance(n, m, state1, state2):
    row1, col1 = divmod(state1, m)
    row2, col2 = divmod(state2, m)
    distance = abs(row1 - row2) + abs(col1 - col2)
    return distance


def generate_and_visit(m, n, visited_states):
    stack = list(range(n * m))
    stack = [state for state in stack if state not in visited_states]
    return stack


def update_neighbors_adjacency_matrix(grid_shape, state, adj_matrix, walls):
    rows, cols = grid_shape
    updated_adjacency_matrix = copy.deepcopy(adj_matrix)

    row = state // cols
    col = state % cols

    neighbors = []
    if row > 0:
        neighbors.append(state - cols)
    if row < rows - 1:
        neighbors.append(state + cols)
    if col > 0:
        neighbors.append(state - 1)
    if col < cols - 1:
        neighbors.append(state + 1)

    for neighbor in neighbors:
        if (state, neighbor) in walls or (neighbor, state) in walls:
            updated_adjacency_matrix[state][neighbor] = 0
            updated_adjacency_matrix[neighbor][state] = 0
        else:
            updated_adjacency_matrix[state][neighbor] = 1
            updated_adjacency_matrix[neighbor][state] = 1

    return updated_adjacency_matrix


def PA_values(m, n, product_nodes, adj_matrix):
    value_table = {}
    for PA_nodes in product_nodes:
        value_table[PA_nodes] = {}
        valid_actions = get_valid_actions(m, n, PA_nodes[0], adj_matrix)
        for action in valid_actions:
            value_table[PA_nodes][action] = 0.0
    for PA_nodes in product_nodes:
        valid_actions = get_valid_actions(m, n, PA_nodes[0], adj_matrix)
        for action in valid_actions:
            if PA_nodes[1] == 'accept_all' or PA_nodes[1] == 'Trash':
                value_table[PA_nodes][action] = 0
    return value_table


def evaluate_label_expression(label_expr, true_label_expr):
    label_conditions = re.split(r'\s*&&\s*', label_expr)
    true_label_conditions = re.split(r'\s*&&\s*', true_label_expr)

    label_set = set()
    true_label_set = set()
    for cond in label_conditions:
        negated = cond.startswith('!')
        label = cond[1:] if negated else cond
        label_set.add((label, negated))

    for cond in true_label_conditions:
        negated = cond.startswith('!')
        label = cond[1:] if negated else cond
        true_label_set.add((label, negated))

    return label_set == true_label_set


def update_trigger(states, true_labels, state_info):
    num_states = len(states)
    num_labels = max(len(info) for info in state_info.values())

    matrix = np.zeros((num_states, num_labels))

    for i, state in enumerate(states):
        for j, (probability, label_expr) in enumerate(state_info[state]):
            is_true_label = evaluate_label_expression(label_expr, true_labels[state])
            matrix[i, j] = abs(1 - probability) if is_true_label else abs(0 - probability)

    inf_norm = np.linalg.norm(matrix, np.inf)
    return inf_norm


def get_valid_actions(m, n, z, adj_matrix):
    row, col = divmod(z, n)
    valid_actions = []

    def state_num(row, col):
        return row * n + col

    if row > 0 and adj_matrix[z][state_num(row - 1, col)] > 0:
        valid_actions.append('up')
    if row < m - 1 and adj_matrix[z][state_num(row + 1, col)] > 0:
        valid_actions.append('down')
    if col > 0 and adj_matrix[z][state_num(row, col - 1)] > 0:
        valid_actions.append('left')
    if col < n - 1 and adj_matrix[z][state_num(row, col + 1)] > 0:
        valid_actions.append('right')

    valid_actions.append('stay')
    return valid_actions


def Value_iteration(m, n, value_table, transition_dict, transitions, product_nodes, gamma, adj_matrix, epsilon):
    a = 1
    b = 0
    r = 1
    c = -1
    transition_map = build_transition_map(transitions)
    max_values = {}
    max_actions = {}
    value_table_2 = copy.deepcopy(value_table)

    iteration = 0
    while True:
        delta = 0

        for state, actions in value_table_2.items():
            max_value = max(actions.values())
            max_action = max(actions, key=actions.get)
            max_actions[state] = max_action
            max_values[state] = max_value

        for i in value_table_2:
            if i[1] != 'accept_all' and i[1] != 'Trash':
                current_physical_state = i[0]
                valid_actions = get_valid_actions(m, n, current_physical_state, adj_matrix)

                for action in valid_actions:
                    expected_reward = 0
                    next_physical_state = get_next_state(m, n, current_physical_state, action, adj_matrix)

                    if next_physical_state is not None:
                        possible_transitions = transition_map.get((i, next_physical_state), [])

                        for j in possible_transitions:
                            first_st = str(j[0])
                            second_st = str(j[1])
                            prob = transition_dict[(first_st, second_st)]
                            phy_transition_value = adj_matrix[current_physical_state][next_physical_state]

                            if j[1][1] == 'accept_all':
                                reward = a * c + b * r
                                partial_reward = prob * (
                                    (phy_transition_value * reward)
                                    + (1 - phy_transition_value) * value_table_2[j[0]]['stay']
                                )
                            elif j[1][1] == 'Trash':
                                reward = a * c / (1 - gamma)
                                partial_reward = prob * (
                                    (phy_transition_value * reward)
                                    + (1 - phy_transition_value) * value_table_2[j[0]]['stay']
                                )
                            else:
                                reward = a * c
                                partial_reward = prob * (
                                    (phy_transition_value * (reward + max_values[j[1]] * gamma))
                                    + (1 - phy_transition_value) * value_table_2[j[0]]['stay']
                                )

                            expected_reward += partial_reward

                    delta = max(delta, abs(value_table_2[i][action] - expected_reward))
                    value_table_2[i][action] = expected_reward

        if delta < epsilon:
            break
        iteration += 1

    for state, actions in value_table_2.items():
        max_value = max(actions.values())
        keys_with_max_value = [k for k, v in actions.items() if v == max_value]
        max_action = random.choice(keys_with_max_value)
        max_actions[state] = max_action

    for state, actions in value_table_2.items():
        max_value = max(actions.values())
        max_values[state] = max_value

    return max_actions, max_values



def filter_adj_matrix(adj_matrix, plan_neighbors):
    plan_neighbors = set(plan_neighbors)
    size = len(adj_matrix)
    filtered = [[0] * size for _ in range(size)]

    for i in plan_neighbors:
        for j in plan_neighbors:
            if adj_matrix[i][j] != 0:
                filtered[i][j] = adj_matrix[i][j]

    return filtered


def prune_dict_by_states(data_dict, valid_states):
    valid_states = set(valid_states)
    pruned = {k: v for k, v in data_dict.items() if k[0] in valid_states}
    return pruned


def prune_transitions_by_states(transitions, valid_states):
    valid_states = set(valid_states)
    pruned = [
        t for t in transitions
        if t[0][0] in valid_states and t[1][0] in valid_states
    ]
    return pruned


def find_paths_in_visited(n, m, current_state, discovered_labels):
    all_path_states = set()
    paths = {}

    def bfs(start, goal):
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            s, path = queue.popleft()
            if s == goal:
                return path
            for nb in get_states_within_h_distance(m, n, s, 1):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, path + [nb]))
        return []

    for label in discovered_labels:
        path = bfs(current_state, label)
        if path:
            paths[label] = path
            all_path_states.update(path)

    return paths, list(all_path_states)
