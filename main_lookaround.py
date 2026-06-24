"""
Alternative planner loop: BEFORE each move, the robot rotates to face every
one-hop neighbor cell that hasn't been observed yet, runs the detector
in that direction, and records the label. Once all neighbors are observed
the planner picks an action and the robot drives one cell.

Reasoning: in the default main.py the robot only observes the cell it is
entering. That's cheap (no extra rotations) but the planner has to commit
to a direction before it knows what's there. With the look-around step
the planner sees the immediate neighborhood first, then commits — at the
cost of up to four extra rotations per move.

Observations are STICKY: once a cell has a non-empty label, we never
re-observe it. The world is static, so a confirmed marker stays.

If the detected distance exceeds ASSIGN_DIST_M the marker is too far to
be in the directly-faced cell — the same soft-hint mechanism from main.py
is used to nudge the belief at the inferred farther cell.
"""

import signal
import sys
import time
import numpy as np

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
from turtle_driver import TurtleBot
from label_detector import LabelDetector

CELL_SIZE_M     = 0.5
ASSIGN_DIST_M   = 0.65
SOFT_MAX_CELLS  = 3
SOFT_P_LABEL    = 0.5
SOFT_P_EMPTY    = 0.5
SOFT_ZETA       = 0.02

# How long to dwell after the rotation finishes before reading detect().
# Gives the grab thread time to land a few frames in the new heading.
LOOKAROUND_DWELL_S = 1.0

# Directions to look around, in the order the robot will face them.
# Choosing an order that minimizes rotation given the robot's current
# heading is possible but not implemented here.
LOOKAROUND_DIRECTIONS = ['right', 'down', 'left', 'up']


def soft_update_belief(belief, state, label,
                       p_label=SOFT_P_LABEL,
                       p_empty=SOFT_P_EMPTY,
                       zeta=SOFT_ZETA):
    raw = []
    for _prob, lbl in belief[state]:
        if lbl == label:
            raw.append((p_label, lbl))
        elif lbl == EMPTY_LABEL:
            raw.append((p_empty, lbl))
        else:
            raw.append((zeta, lbl))
    total = sum(p for p, _ in raw)
    belief[state] = [(p / total, lbl) for p, lbl in raw]
    return belief


# ───────────────────────── grid / DFA setup ──────────────────────────
n, m = 6, 3
p_h = 4
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

product_graph, transitions, product_nodes, PR_adj_matrix = generate_product_automaton(
    nodes, edges, adj_org, dfa_states, dfa_transitions, observations
)
transitions = list(dict.fromkeys(transitions))

initial_belief = {
    3: {'a && !b && !c': 0.8, '!a && !b && !c': 0.2},
    4: {'!a && b && !c': 0.8, '!a && !b && !c': 0.2},
    9: {'!a && !b && c': 0.8, '!a && !b && !c': 0.2},
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

plan_neighbors = get_states_within_h_distance(n, m, current_physical_state, p_h)
adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)
pruned_set = prune_dict_by_states(PA_values(n, m, product_nodes, adj_matrix), plan_neighbors)
portion_transitions = prune_transitions_by_states(transitions, plan_neighbors)
transition_dict = probabilistic_labeling_next(portion_transitions, observation_probabilities, dfa_transitions, adj_matrix)
policy, all_values = Value_iteration(n, m, pruned_set, transition_dict, portion_transitions, product_nodes, gamma, adj_matrix, epsilon)

visited_states = [0]
visited_states_un = [0]
previous_probabilities = {}
perceived_labels = {0: EMPTY_LABEL}   # robot starts at cell 0
full_traj = []
full_physical_traj = []
discovered_labels = []
current_value = all_values[current_state]
p_t_t, p_t_c = 0, 0
counter, j = 0, 0
step_count = 0

bot = TurtleBot()
detector = LabelDetector(camera_index=0, record_path="run_lookaround.mp4")


def _shutdown_handler(signum, _frame):
    print(f"\n[interrupt] signal {signum} received, shutting down...")
    try: bot.shutdown()
    except Exception: pass
    try: detector.close()
    except Exception: pass
    sys.exit(0)


signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


def physical_neighbor(state, direction):
    """Return the cell index reached from `state` by moving in `direction`,
    or None if that direction leaves the grid."""
    return get_next_state(n, m, state, direction, adj_org)


def observe_cell_in_direction(cell_we_face, direction):
    """Rotate to face `direction`, dwell briefly, read detector, apply
    sticky labeling + soft hints. Returns nothing — side-effects on the
    global `belief` and `perceived_labels`."""
    global belief, observation_probabilities

    # Already labelled? Skip (sticky).
    prior = perceived_labels.get(cell_we_face)
    if prior is not None and prior != EMPTY_LABEL:
        print(f"  [LOOK] cell {cell_we_face} already known as {prior!r}; skip.")
        return

    print(f"  [LOOK] facing {direction:<5} -> cell {cell_we_face}")
    detector.reset_observation_window()
    bot.face(direction)
    time.sleep(LOOKAROUND_DWELL_S)
    detected_label, detected_dist, detected_color, snapshot = detector.detect()

    if snapshot:
        seen = ", ".join(
            f"{c}={d*100:.1f}cm" for c, d in sorted(snapshot.items(), key=lambda kv: kv[1])
        )
        print(f"         [DETECT] {seen}")

    soft_target_cell  = None
    soft_target_label = None

    if (detected_color is not None
            and detected_label is not None
            and detected_dist < ASSIGN_DIST_M):
        this_obs = detected_label
    else:
        this_obs = EMPTY_LABEL
        # Far-but-mapped colour → soft hint at a cell further down `direction`.
        if detected_color is not None and detected_label is not None:
            cell_offset = int(round(detected_dist / CELL_SIZE_M))
            cell_offset = max(1, min(cell_offset, SOFT_MAX_CELLS))
            target = cell_we_face
            # cell_we_face is already 1 hop away; for cell_offset=1 the
            # target IS cell_we_face. For larger offsets, step further.
            for _ in range(cell_offset - 1):
                nxt = get_next_state(n, m, target, direction, adj_org)
                if nxt is None:
                    target = None
                    break
                target = nxt
            if (target is not None
                    and target != cell_we_face
                    and perceived_labels.get(target) in (None, EMPTY_LABEL)):
                soft_target_cell  = target
                soft_target_label = detected_label

    # Sticky assignment for cell_we_face.
    if prior is None or prior == EMPTY_LABEL:
        perceived_labels[cell_we_face] = this_obs
        if this_obs != EMPTY_LABEL:
            print(f"         [LABEL] cell {cell_we_face} -> {this_obs}")
            belief = update(belief, cell_we_face, this_obs)
            if cell_we_face not in discovered_labels:
                discovered_labels.append(cell_we_face)
        else:
            note = (f"  + soft hint cell {soft_target_cell}"
                    if soft_target_cell is not None else "")
            print(f"         [LABEL] cell {cell_we_face} empty{note}")

    if soft_target_cell is not None and soft_target_label is not None:
        belief = soft_update_belief(belief, soft_target_cell, soft_target_label)
        print(f"         [SOFT] cell {soft_target_cell} <- "
              f"P({soft_target_label})={SOFT_P_LABEL}, P(empty)={SOFT_P_EMPTY}")

    observation_probabilities = belief

    if cell_we_face not in visited_states_un:
        visited_states_un.append(cell_we_face)


print("=" * 60)
print(f"Starting LOOK-AROUND run | grid {n}x{m} | formula: {formula_str}")
print("=" * 60)

while next_dfa_state != 'accept_all':
    not_visited = generate_and_visit(m, n, visited_states_un)
    if not_visited == [] and current_value < -1 / (1 - gamma) + 100*epsilon:
        break

    current_state = next_state
    current_dfa_state = current_state[1]
    current_physical_state = current_state[0]
    full_traj.append(current_state)

    # ─── Look around: face each unmapped one-hop neighbor and observe ──
    step_count += 1
    print(f"\n[Step {step_count:>3}] at cell {current_physical_state} | dfa={current_dfa_state}")
    for direction in LOOKAROUND_DIRECTIONS:
        neighbor = physical_neighbor(current_physical_state, direction)
        if neighbor is None:
            continue   # off-grid in that direction
        if neighbor == current_physical_state:
            continue   # 'stay' shouldn't be in the list, but defensive
        prior = perceived_labels.get(neighbor)
        if prior is not None and prior != EMPTY_LABEL:
            # Already confirmed non-empty — sticky, no need to re-look.
            continue
        observe_cell_in_direction(neighbor, direction)

    # ─── Re-plan with the freshly observed neighborhood ──────────────
    plan_neighbors = get_states_within_h_distance(n, m, current_physical_state, p_h)
    adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)
    pruned_set = prune_dict_by_states(PA_values(n, m, product_nodes, adj_matrix), plan_neighbors)
    portion_transitions = prune_transitions_by_states(transitions, plan_neighbors)
    transition_dict = probabilistic_labeling_next(
        portion_transitions, observation_probabilities, dfa_transitions, adj_matrix
    )
    _t = time.time()
    policy, all_values = Value_iteration(
        n, m, pruned_set, transition_dict, portion_transitions, product_nodes, gamma, adj_matrix, epsilon
    )
    p_t_t += time.time() - _t
    p_t_c += 1

    # Expand horizon while value is below the unreachable threshold.
    current_value_0 = all_values[current_state]
    if current_value_0 > -1 / (1 - gamma) + 100*epsilon:
        p_h = initial_p_h
    while current_value_0 < -1 / (1 - gamma) + 100*epsilon:
        p_h += 1
        counter += 1
        print(f"  [REPLAN-EXPAND #{counter}] value {current_value_0:.2f} < LOW; p_h -> {p_h}")
        plan_neighbors = get_states_within_h_distance(n, m, current_physical_state, p_h)
        adj_matrix = filter_adj_matrix(adj_org, plan_neighbors)
        pruned_set = prune_dict_by_states(PA_values(n, m, product_nodes, adj_matrix), plan_neighbors)
        portion_transitions = prune_transitions_by_states(transitions, plan_neighbors)
        transition_dict = probabilistic_labeling_next(
            portion_transitions, observation_probabilities, dfa_transitions, adj_matrix
        )
        policy, all_values = Value_iteration(
            n, m, pruned_set, transition_dict, portion_transitions, product_nodes, gamma, adj_matrix, epsilon
        )
        current_value_0 = all_values[current_state]

    current_value = current_value_0
    action = policy[current_state]
    next_physical_state = get_next_state(n, m, current_physical_state, action, adj_matrix)
    print(f"  [DECIDE] action={action:<5} -> cell {next_physical_state} | value={current_value:8.2f}")

    if action == 'stay' or next_physical_state is None:
        print(f"  [WARN] action=stay at cell {current_physical_state}; stopping.")
        break

    # ─── Execute one cell move ──────────────────────────────────────
    # IMPORTANT: wait for the whole move to finish before the next iteration,
    # otherwise the next look-around's bot.face() runs mid-drive in parallel
    # with the still-running _drive_continuous and the robot turns half-way
    # through. wait_for_cell_entry() only returns at the half-cell mark —
    # we need bot.wait() which blocks until the motion thread is idle.
    bot.move(action)
    bot.wait()

    # Mark cell as visited.
    if next_physical_state not in visited_states_un:
        visited_states_un.append(next_physical_state)
    visited_states.append(current_physical_state)
    full_physical_traj.append(current_physical_state)

    # Update DFA based on the cell just entered (whose label we observed
    # during the look-around at the previous step, or that we'll observe
    # next iteration if this is the first time we entered it).
    label = perceived_labels.get(next_physical_state, EMPTY_LABEL)
    for tr in dfa_transitions:
        if tr[0] == current_dfa_state and label == tr[1][0]:
            next_dfa_state = tr[2]
    next_state = (next_physical_state, next_dfa_state)

    print(f"           entered cell {next_physical_state}; next_dfa={next_dfa_state}")

bot.wait()
detector.close()

print()
print("=" * 60)
print("Run complete")
print("=" * 60)
print(f"  Total steps        : {step_count}")
print(f"  Trajectory length  : {len(full_physical_traj)}")
print(f"  Replans            : {p_t_c}")
if p_t_c:
    print(f"  Avg replan time    : {p_t_t/p_t_c:.3f}s")
print(f"  Perceived labels   : {perceived_labels}")
