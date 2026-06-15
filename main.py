import signal
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import spot
import buddy

from grid import create_graph
from labeling import (
    get_states_within_h_distance, assign_probabilities_g3, update
)

EMPTY_LABEL = '!a && !b && !c'
from dfa import extract_atomic_props, extract_dfa_transitions_with_trash_expanded
from product_automaton import generate_product_automaton
from planning import (
    PA_values, Value_iteration, generate_and_visit, get_next_state,
    update_trigger, filter_adj_matrix, prune_dict_by_states, prune_transitions_by_states, find_paths_in_visited
)
from grid import grid_probabilities
from visualization import generate_grid_environment
from turtle_driver import TurtleBot
from label_detector import LabelDetector

CELL_SIZE_M = 0.3  # must match TurtleBot.CELL_SIZE

n, m = 4, 4
p_h = 3
initial_p_h = p_h
policy_p_h = p_h
threshold = 0
gamma = 0.99
epsilon = 0.01
formula_str = "F((a & F((b & F(c)))))"

start_time = time.time()
nodes, edges, adj_matrix_np = create_graph(n, m)
adj_org = adj_matrix_np.tolist()

atomic_props = extract_atomic_props(formula_str)
dfa_transitions, initial_state, trash_states_set = extract_dfa_transitions_with_trash_expanded(formula_str)
dfa_states = list({t[0] for t in dfa_transitions} | {t[2] for t in dfa_transitions})

observations = list(set(cond for _, conds, _ in dfa_transitions for cond in conds))

time_ps = time.time()
product_graph, transitions, product_nodes, PR_adj_matrix = generate_product_automaton(
    nodes, edges, adj_org, dfa_states, dfa_transitions, observations
)
print(f"Product automaton construction time: {time.time() - time_ps:.3f}s")

transitions = list(dict.fromkeys(transitions))

initial_belief = {
    2:  {'a && !b && !c': 0.8, '!a && !b && !c': 0.2},   # cell 2  expects red    (a)
    10: {'!a && b && !c': 0.8, '!a && !b && !c': 0.2},   # cell 10 expects yellow (b)
    15: {'!a && !b && c': 0.8, '!a && !b && !c': 0.2},   # cell 15 expects green  (c)
}
belief = assign_probabilities_g3(n, m, atomic_props, initial_belief=initial_belief)
observation_probabilities = belief

initial_state = str(initial_state)
start_node = (0, initial_state)
current_state = start_node
next_state = start_node
next_dfa_state = initial_state
current_physical_state = 0

from dfa import probabilistic_labeling_next

plan_neighbors = get_states_within_h_distance(m, n, current_physical_state, p_h)
adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)
pruned_set = prune_dict_by_states(PA_values(m, n, product_nodes, adj_matrix), plan_neighbors)
portion_transitions = prune_transitions_by_states(transitions, plan_neighbors)
transition_dict = probabilistic_labeling_next(portion_transitions, observation_probabilities, dfa_transitions, adj_matrix)
_t = time.time()
policy, all_values = Value_iteration(m, n, pruned_set, transition_dict, portion_transitions, product_nodes, gamma, adj_matrix, epsilon)
print(f"Initial policy computed in {(time.time() - _t)*1000:.1f} ms")

visited_states = [0]
visited_states_un = [0]   # only cells the robot has entered OR camera-observed
previous_probabilities = {}
perceived_labels = {}        # cell_index -> last label the camera detected there
full_traj = []
full_physical_traj = []
discovered_labels = []
current_value = all_values[current_state]
p_t_t, p_t_c = 0, 0
counter, j = 0, 0
step_count = 0

bot = TurtleBot()
detector = LabelDetector(camera_index=0, record_path="run.mp4")

def _shutdown_handler(signum, _frame):
    # Override rospy's SIGINT handler so a single Ctrl-C reliably stops us
    # *and* finalizes the video file (atexit also fires after sys.exit).
    print(f"\n[interrupt] signal {signum} received, shutting down...")
    try: bot.shutdown()
    except Exception: pass
    try: detector.close()
    except Exception: pass
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

print("=" * 60)
print(f"Starting run | grid {n}x{m} | formula: {formula_str}")
print("=" * 60)

while next_dfa_state != 'accept_all':
    not_visited = generate_and_visit(m, n, visited_states_un)
    if not_visited == [] and current_value < -1 / (1 - gamma) + 100*epsilon:
        break

    current_state = next_state
    current_dfa_state = current_state[1]
    current_physical_state = current_state[0]
    full_traj.append(current_state)
    action = policy[current_state]
    current_value = all_values[current_state]

    step_count += 1
    next_physical_state = get_next_state(m, n, current_physical_state, action, adj_matrix)
    print(
        f"\n[Step {step_count:>3}] "
        f"state={current_physical_state:>3} -> {next_physical_state:<3} "
        f"| action={action:<5} "
        f"| dfa={current_dfa_state} "
        f"| value={current_value:8.2f}"
    )

    bot.move(action)
    bot.wait_for_cell_entry()
    # Let the heading finish settling before reading the camera, so the FOV
    # matches the action direction (and the marker isn't seen mid-turn).
    bot.wait_for_heading_settled()

    detected_label, detected_dist, detected_color = detector.detect()
    assigned_cell = None
    # The cell directly in front of the robot — what the camera is looking at.
    facing_cell = get_next_state(m, n, next_physical_state, action, adj_org)

    if detected_color is not None:
        # Only trust a detection if the marker is within one cell of the
        # camera (closer than CELL_SIZE_M). Anything farther is too noisy to
        # assign reliably.
        if detected_dist < CELL_SIZE_M and facing_cell is not None:
            assigned_cell = facing_cell
            dfa_tag = f"-> {detected_label}" if detected_label is not None else "(unmapped)"
            print(
                f"  [LABEL] detected {detected_color} @ {detected_dist*100:5.1f} cm "
                f"-> cell {assigned_cell} {dfa_tag}"
            )
        else:
            print(
                f"  [LABEL] detected {detected_color} @ {detected_dist*100:5.1f} cm "
                f"(too far, not assigned)"
            )

    # Record the observation about the facing cell. A mapped marker within
    # range overwrites with its label; anything else (no detection, unmapped
    # colour, or out-of-range) is treated as "observed empty" so the trigger
    # can react to "expected a label, saw none".
    if facing_cell is not None:
        if assigned_cell == facing_cell and detected_label is not None:
            perceived_labels[facing_cell] = detected_label
        else:
            perceived_labels[facing_cell] = EMPTY_LABEL

    current_value_0 = all_values[current_state]
    plan_neighbors = get_states_within_h_distance(m, n, next_physical_state, p_h)

    adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)

    if current_value_0 > -1 / (1 - gamma) + 100*epsilon:
        p_h = initial_p_h
    while current_value_0 < -1 / (1 - gamma)+ 100*epsilon:
        p_h += 1
        counter += 1
        print(f"  [replan #{counter}] p_h={p_h} | value before: {current_value_0:8.2f}")
        plan_neighbors = get_states_within_h_distance(m, n, next_physical_state, p_h)

        paths, new_states_to_add = find_paths_in_visited(n, m, next_physical_state, discovered_labels)
        for state in new_states_to_add:
            if state not in plan_neighbors:
                plan_neighbors.append(state)

        adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)

        initial_PA_values = PA_values(m, n, product_nodes, adj_matrix)
        pruned_set = prune_dict_by_states(initial_PA_values, plan_neighbors)
        policy_p_h = p_h

        portion_transitions = prune_transitions_by_states(transitions, plan_neighbors)
        transition_dict = probabilistic_labeling_next(
            portion_transitions, observation_probabilities, dfa_transitions, adj_matrix
        )
        start_time_3 = time.time()
        policy, all_values = Value_iteration(
            m, n, pruned_set, transition_dict, portion_transitions, product_nodes, gamma, adj_matrix, epsilon
        )
        end_time_3 = time.time()
        p_t_i = end_time_3 - start_time_3
        p_t_t += p_t_i
        p_t_c += 1
        current_value_0 = all_values[current_state]
        print(f"  [replan #{counter}] policy computed in {p_t_i*1000:.1f} ms | value after: {current_value_0:8.2f}")

    # Mark as explored: the cell the robot has been in, the cell it just
    # entered, and the cell it is facing (which we observed this iteration).
    # No Manhattan neighbourhood claim — the forward-facing camera can't see
    # sideways.
    for s in (current_physical_state, next_physical_state, facing_cell):
        if s is not None and s not in visited_states_un:
            visited_states_un.append(s)
    visited_states.append(current_physical_state)
    full_physical_traj.append(current_physical_state)

    # The facing cell is observed every step (with EMPTY_LABEL if nothing
    # was seen) so the trigger can react to belief-vs-reality disagreements.
    just_observed = [facing_cell] if facing_cell is not None else []

    previous_probabilities = {}
    neighbor_true_labels   = {}
    for state in just_observed:
        previous_probabilities[state] = belief[state]
        neighbor_true_labels[state]   = perceived_labels[state]

    previous_probabilities = {k: v.tolist() for k, v in previous_probabilities.items()}
    if just_observed:
        trigger_function_value = update_trigger(
            just_observed, neighbor_true_labels, previous_probabilities
        )
    else:
        trigger_function_value = 0.0

    for state in just_observed:
        neighbor_label = perceived_labels[state]
        belief = update(belief, state, neighbor_label)
        if neighbor_label != EMPTY_LABEL and state not in discovered_labels:
            discovered_labels.append(state)

    label = perceived_labels.get(next_physical_state, EMPTY_LABEL)

    for i in dfa_transitions:
        if i[0] == current_dfa_state and label == i[1][0]:
            next_dfa_state = i[2]

    next_state = (next_physical_state, next_dfa_state)

    next_value = all_values[next_state]
    label_str = label if label != EMPTY_LABEL else '-'
    print(f"           next_dfa={next_dfa_state} | label={label_str} | next_value={next_value:8.2f}")
    # Only sharpen belief on the just-entered cell if the camera actually
    # saw something for it; otherwise let the prior stand.
    if next_physical_state in perceived_labels:
        belief = update(belief, next_physical_state, perceived_labels[next_physical_state])
    observation_probabilities = belief

    j += 1
    if trigger_function_value > threshold or next_value == current_value or j >= policy_p_h - 1:
        j = 0
        counter += 1
        print(f"  [replan #{counter}] p_h={p_h} | value: {next_value:8.2f}")

        paths, new_states_to_add = find_paths_in_visited(n, m, next_physical_state, discovered_labels)
        for state in new_states_to_add:
            if state not in plan_neighbors:
                plan_neighbors.append(state)

        adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)

        initial_PA_values = PA_values(m, n, product_nodes, adj_matrix)
        pruned_set = prune_dict_by_states(initial_PA_values, plan_neighbors)
        policy_p_h = p_h

        portion_transitions = prune_transitions_by_states(transitions, plan_neighbors)
        transition_dict = probabilistic_labeling_next(
            portion_transitions, observation_probabilities, dfa_transitions, adj_matrix
        )
        start_time_3 = time.time()
        policy, all_values = Value_iteration(
            m, n, pruned_set, transition_dict, portion_transitions, product_nodes, gamma, adj_matrix, epsilon
        )
        end_time_3 = time.time()
        p_t_i = end_time_3 - start_time_3
        p_t_t += p_t_i
        p_t_c += 1
        print(f"  [replan #{counter}] policy computed in {p_t_i*1000:.1f} ms")

        if current_physical_state not in visited_states_un:
            visited_states_un.append(current_physical_state)

bot.wait()
detector.close()
full_physical_traj.append(next_physical_state)
full_traj.append(next_state)
probabilities = grid_probabilities(belief, n, m)

print()
print("=" * 60)
print("Run complete")
print("=" * 60)
print(f"  Total steps        : {step_count}")
print(f"  Trajectory length  : {len(full_physical_traj)}")
print(f"  Replans            : {p_t_c}")
if p_t_c:
    print(f"  Avg replan time    : {p_t_t/p_t_c:.3f}s")
print(f"  Total time         : {time.time() - start_time:.3f}s")
print(f"  Trajectory         : {full_physical_traj}")

generate_grid_environment(n, m, full_physical_traj, probabilities)
plt.show()
