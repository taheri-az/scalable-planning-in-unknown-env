import re
import copy
import pickle
import itertools
import numpy as np
from collections import deque


def check_label_l(state_number):
    if state_number == 62:
        return "a && !b && !c && !d"
    elif state_number == 9:
        return "!a && b && !c && !d"
    elif state_number == 47:
        return "!a && !b && c && !d"
    elif state_number == 40:
        return "!a && !b && !c && d"
    else:
        return "!a && !b && !c && !d"




def get_states_within_h_distance(m, n, current_state, h):
    def state_to_row_col(state):
        return divmod(state, n)

    def row_col_to_state(row, col):
        if 0 <= row < m and 0 <= col < n:
            return row * n + col
        return None

    def get_adjacent_states(state):
        row, col = state_to_row_col(state)
        adjacent_states = []
        for r, c in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            adjacent_state = row_col_to_state(r, c)
            if adjacent_state is not None:
                adjacent_states.append(adjacent_state)
        return adjacent_states

    visited = set()
    queue = deque([(current_state, 0)])
    while queue:
        state, distance = queue.popleft()
        if distance > h:
            continue
        if state in visited:
            continue
        visited.add(state)
        for next_state in get_adjacent_states(state):
            if next_state not in visited:
                queue.append((next_state, distance + 1))

    return list(visited)


def get_neighbors(cell, n, m):
    x, y = divmod(cell, m)
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < m:
            neighbors.append(nx * m + ny)
    return neighbors


def generate_conditions(regions):
    num_regions = len(regions)
    ordered_subsets = []

    for i in range(num_regions):
        single_positive = [0] * num_regions
        single_positive[i] = 1
        ordered_subsets.append(tuple(single_positive))

    for count in range(2, num_regions + 1):
        for subset in itertools.combinations(range(num_regions), count):
            combination = [0] * num_regions
            for index in subset:
                combination[index] = 1
            ordered_subsets.append(tuple(combination))

    ordered_subsets.append(tuple([0] * num_regions))

    ordered_conditions = [
        ' && '.join([f'{regions[k]}' if subset[k] == 1 else f'!{regions[k]}' for k in range(num_regions)])
        for subset in ordered_subsets
    ]
    return ordered_conditions


def generate_observations(regions):
    num_regions = len(regions)
    observations = []

    region_subsets = list(itertools.product([0, 1], repeat=num_regions))

    for region in regions:
        observations.append(region)
        observations.append(f'!{region}')

    for subset in region_subsets:
        if sum(subset) == 0:
            observations.append('1')
            continue
        condition = ' && '.join(
            [f'{regions[i]}' if subset[i] == 1 else f'!{regions[i]}' for i in range(num_regions)]
        )
        observations.append(condition)

    return observations


def assign_probabilities_g3(n, m, regions, initial_belief=None, zeta=0.002):
    """
    Build initial belief grid for all 2^len(regions) label combinations.

    For every state x and every label l with probability 0, the probability is
    raised to zeta before renormalization, ensuring the robot has a non-zero
    incentive to visit every state and verify every label.

    Args:
        n, m: grid dimensions.
        regions: atomic proposition names, e.g. ['a', 'b', 'c', 'd'].
        initial_belief: optional {state_index: {label_string: probability}}.
                        States not listed receive an empty-biased prior.
                        Labels missing from a state's dict are treated as 0.
        zeta: small constant applied to zero-probability labels before renormalization.
    """
    labels = [
        ' && '.join(regions[i] if bits[i] else f'!{regions[i]}' for i in range(len(regions)))
        for bits in itertools.product([0, 1], repeat=len(regions))
    ]
    num_labels = len(labels)
    empty_label = ' && '.join(f'!{r}' for r in regions)

    grid = np.empty((n * m, num_labels), dtype=object)
    for state in range(n * m):
        if initial_belief is not None and state in initial_belief:
            state_dist = initial_belief[state]
            probs = np.array([state_dist.get(label, 0.0) for label in labels], dtype=float)
        else:
            default_mass = max(0.0, 1.0 - zeta * (num_labels - 1))
            probs = np.array(
                [default_mass if label == empty_label else zeta for label in labels],
                dtype=float,
            )
        probs[probs == 0.0] = zeta
        probs /= probs.sum()

        grid[state] = [(float(probs[j]), labels[j]) for j in range(num_labels)]
    return grid


def update(grid, state, label):
    label_numbers = len(grid[0])
    for i, (probability, lbl) in enumerate(grid[state]):
        if lbl == label:
            grid[state][i] = (1, lbl)
        else:
            grid[state][i] = (0 / (label_numbers - 1), lbl)
    return grid





def convert_belief_map_to_logical_grid(pkl_file_path, n, m, save_to=None):
    label_to_formula = {
        frozenset(['a']): 'a && !b && !c',
        frozenset(['b']): '!a && b && !c',
        frozenset(['c']): '!a && !b && c',
        frozenset(['a', 'b']): 'a && b && !c',
        frozenset(['a', 'c']): 'a && !b && c',
        frozenset(['b', 'c']): '!a && b && c',
        frozenset(['a', 'b', 'c']): 'a && b && c',
        frozenset(): '!a && !b && !c',
    }

    with open(pkl_file_path, 'rb') as f:
        node_dict = pickle.load(f)

    num_states = n * m
    grid = np.empty((num_states, 8), dtype=object)

    ordered_labels = [
        frozenset(['a']),
        frozenset(['b']),
        frozenset(['c']),
        frozenset(['a', 'b']),
        frozenset(['a', 'c']),
        frozenset(['b', 'c']),
        frozenset(['a', 'b', 'c']),
        frozenset(),
    ]

    for state in range(num_states):
        belief = node_dict.get(state, {frozenset(): 1.0})
        row = [(belief.get(label, 0.0), label_to_formula[label]) for label in ordered_labels]
        grid[state] = row

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(grid, f)

    return grid


