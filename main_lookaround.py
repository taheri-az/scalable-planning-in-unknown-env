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

import os
import signal
import sys
import time
import pickle
import argparse
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

# Trigger threshold: replan only when update_trigger() across the cells
# observed in this step's look-around strictly exceeds this. 0.0 means
# any non-trivial belief change triggers a replan.
TRIGGER_THRESHOLD = 0.0

# The four cardinal directions. The order used at each step is chosen
# dynamically based on the robot's current yaw — see
# `lookaround_order_from_yaw` below.
ALL_DIRECTIONS = ['right', 'down', 'left', 'up']


def lookaround_order_from_yaw(current_yaw, candidates, bot):
    """Sort `candidates` (a subset of ALL_DIRECTIONS) into an order that
    minimizes cumulative rotation when visiting them in sequence, starting
    from `current_yaw`.

    Greedy: at each step pick the unvisited direction whose target yaw is
    closest to the robot's current (simulated) yaw. Updates the simulated
    yaw after each pick. For 4-cardinal directions this is optimal — the
    greedy choice corresponds to walking around the compass in the cheaper
    rotational direction without doubling back."""
    remaining = list(candidates)
    order = []
    sim_yaw = current_yaw
    while remaining:
        # Pick the direction whose target yaw is closest to sim_yaw.
        def yaw_cost(d):
            target = bot._action_target_yaw(d)
            return bot._yaw_diff(target, sim_yaw)
        nxt = min(remaining, key=yaw_cost)
        order.append(nxt)
        sim_yaw = bot._action_target_yaw(nxt)
        remaining.remove(nxt)
    return order


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


def canonical_label(label):
    terms = [t.strip() for t in label.split("&&")]
    terms.sort()
    return " && ".join(terms)

# ───────────────────────── CLI / belief source ───────────────────────
_argp = argparse.ArgumentParser(description="Look-around LTL planner")
_argp.add_argument("--belief", default="belief.pkl",
                   help="path to a belief pickle from build_belief_gui.py "
                        "(default: belief.pkl; falls back to the hard-coded "
                        "prior if the file is absent)")
_argp.add_argument("--no-viz", action="store_true",
                   help="disable the live grid visualization window")
_cli, _ = _argp.parse_known_args()

# ───────────────────────── grid / DFA setup ──────────────────────────
n, m = 6, 3
p_h = 4
initial_p_h = p_h
policy_p_h = p_h
threshold = 0
gamma = 0.99
epsilon = 0.01
# formula_str = "F((a & F((b & F(c)))))"
formula_str = "F a & F b & !c U a & !c U b"

start_time = time.time()
nodes, edges, adj_matrix_np = create_graph(n, m)
adj_org = adj_matrix_np.tolist()

atomic_props = extract_atomic_props(formula_str)

# ── Initial belief: load from a pickle built by build_belief_gui.py if one
# is present, otherwise fall back to the hard-coded prior below. The pickle
# carries its own (n, m, atomics); if they disagree with this run's formula
# we warn and ignore the file rather than silently mis-planning.
_hardcoded_initial_belief = {
    3: {'a && !b && !c': 0.8, '!a && !b && !c': 0.2},
    4: {'!a && b && !c': 0.8, '!a && !b && !c': 0.2},
    9: {'!a && !b && c': 0.8, '!a && !b && !c': 0.2},
}

belief = None
if os.path.exists(_cli.belief):
    try:
        with open(_cli.belief, "rb") as _f:
            _data = pickle.load(_f)
        if (_data["n"], _data["m"]) != (n, m):
            print(f"[belief] {_cli.belief} is for grid "
                  f"{_data['n']}x{_data['m']}, but this run is {n}x{m}; "
                  f"adopting the pickle's grid size.")
            n, m = _data["n"], _data["m"]
            nodes, edges, adj_matrix_np = create_graph(n, m)
            adj_org = adj_matrix_np.tolist()
        if set(_data["atomics"]) != set(atomic_props):
            print(f"[belief] WARNING: pickle atomics {_data['atomics']} != "
                  f"formula atomics {atomic_props}; ignoring pickle.")
        else:
            belief = _data["belief"]
            print(f"[belief] loaded initial belief from {_cli.belief} "
                  f"({len(_data.get('initial_belief', {}))} cells with priors).")
    except Exception as _e:
        print(f"[belief] failed to load {_cli.belief}: {_e}; "
              f"falling back to hard-coded prior.")

if belief is None:
    print("[belief] using hard-coded prior "
          "(run build_belief_gui.py to make belief.pkl).")
    belief = assign_probabilities_g3(n, m, atomic_props,
                                     initial_belief=_hardcoded_initial_belief)

dfa_transitions, initial_state, trash_states_set = extract_dfa_transitions_with_trash_expanded(formula_str)

normalized = []

for src, conds, dst in dfa_transitions:
    normalized.append(
        (
            src,
            [canonical_label(c) for c in conds],
            dst
        )
    )

dfa_transitions = normalized
dfa_states = list({t[0] for t in dfa_transitions} | {t[2] for t in dfa_transitions})
observations = list(set(cond for _, conds, _ in dfa_transitions for cond in conds))
print("\n================ DFA =================")
for src, conds, dst in dfa_transitions:
    for cond in conds:
        print(f"{src:>4} -- {cond:25} --> {dst}")
print("Trash states:", trash_states_set)
print("======================================\n")


product_graph, transitions, product_nodes, PR_adj_matrix = generate_product_automaton(
    nodes, edges, adj_org, dfa_states, dfa_transitions, observations
)
transitions = list(dict.fromkeys(transitions))

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
# Collapse belief at the starting cell to match perceived_labels — if we know
# we're standing on cell 0 and there's no marker on us, belief[0] should be
# a Dirac at empty, not the seeded prior.
belief = update(belief, 0, EMPTY_LABEL)
observation_probabilities = belief
full_traj = []
full_physical_traj = []
discovered_labels = []
current_value = all_values[current_state]
p_t_t, p_t_c = 0, 0
counter, j = 0, 0
step_count = 0

bot = TurtleBot()
detector = LabelDetector(camera_index=0, record_path="run_lookaround.mp4")

# Live grid visualization (robot pose + discovered labels + belief), updated
# as the robot moves. Runs in its own thread; no-op if Tk/display is missing
# or --no-viz was passed.
from live_viz import LiveViz
if _cli.no_viz:
    from live_viz import _NullViz
    viz = _NullViz()
else:
    viz = LiveViz(n, m, atomic_props, belief_pkl=_cli.belief)
# The planner already loaded belief (possibly from _cli.belief); push it so the
# first frame reflects the exact prior after any startup collapse (e.g. cell 0).
viz.update(robot_cell=0, heading=None, perceived=perceived_labels,
           belief=belief, dfa_state=str(initial_state), step=0)


def _shutdown_handler(signum, _frame):
    print(f"\n[interrupt] signal {signum} received, shutting down...")
    try: viz.close()
    except Exception: pass
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
    """Rotate to face `direction`, dwell briefly, read the detector, and
    return what we saw. DOES NOT mutate `belief`. Sticky labels in
    `perceived_labels` are updated here (so the next iteration knows not
    to re-observe), but belief updates are deferred to the caller after
    update_trigger() has been computed against the *prior* belief.

    Returns dict: {
        'cell':        int,   # the cell we observed
        'label':       str,   # the perceived label (may be EMPTY_LABEL)
        'soft_target': int or None,
        'soft_label':  str  or None,
    }
    """
    # Already observed (empty or labelled)? Skip — labels are sticky for
    # non-empty, and we don't waste rotations on cells we've already
    # looked at. World is static; nothing changes between visits.
    prior = perceived_labels.get(cell_we_face)
    if prior is not None:
        print(f"  [LOOK] cell {cell_we_face} already observed as {prior!r}; skip.")
        return None

    print(f"  [LOOK] facing {direction:<5} -> cell {cell_we_face}")
    # Rotate FIRST, then reset the detection window so the grab thread's
    # min-distance buffer only contains frames taken after the camera has
    # stopped moving.
    bot.face(direction)
    detector.reset_observation_window()
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
        this_obs = canonical_label(detected_label)
    else:
        this_obs = EMPTY_LABEL
        # Far-but-mapped colour → soft hint at a cell further down `direction`.
        if detected_color is not None and detected_label is not None:
            cell_offset = int(round(detected_dist / CELL_SIZE_M))
            cell_offset = max(1, min(cell_offset, SOFT_MAX_CELLS))
            target = cell_we_face
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
                soft_target_label = canonical_label(detected_label)

    # Sticky perceived_labels — but DON'T touch belief here.
    perceived_labels[cell_we_face] = this_obs
    if this_obs != EMPTY_LABEL:
        print(f"         [LABEL] cell {cell_we_face} -> {this_obs}")
        if cell_we_face not in discovered_labels:
            discovered_labels.append(cell_we_face)
    else:
        note = (f"  + soft hint cell {soft_target_cell}"
                if soft_target_cell is not None else "")
        print(f"         [LABEL] cell {cell_we_face} empty{note}")

    if cell_we_face not in visited_states_un:
        visited_states_un.append(cell_we_face)

    return {
        'cell':        cell_we_face,
        'label':       this_obs,
        'soft_target': soft_target_cell,
        'soft_label':  soft_target_label,
    }


print("=" * 60)
print(f"Starting LOOK-AROUND run | grid {n}x{m} | formula: {formula_str}")
print("=" * 60)



def _run_planner():
    global action, adj_matrix, all_values, belief, counter
    global current_dfa_state, current_physical_state, current_state
    global current_value, current_value_0, just_observed, label
    global neighbor_true_labels, next_dfa_state, next_physical_state
    global next_state, not_visited, observation_probabilities
    global observations_this_step, ordered, p_h, p_t_c, p_t_t
    global plan_neighbors, policy, portion_transitions
    global previous_probabilities, pruned_set, replan_needed, step_count
    global transition_dict, trigger_function_value

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

        # Pre-filter: only directions whose neighbor cell exists AND is unmapped.
        # Then order them so we visit the closest-to-current-yaw first, etc.,
        # to minimize total rotation. Example: at cell 3 already facing 'down',
        # if we need to observe cells 4 (right) and 6 (down), we visit 6 first
        # (no rotation) then 4 (one 90° turn), instead of doing 4 then 6
        # (two turns).
        candidates = []
        for direction in ALL_DIRECTIONS:
            neighbor = physical_neighbor(current_physical_state, direction)
            if neighbor is None:
                continue
            if neighbor == current_physical_state:
                continue
            if neighbor in perceived_labels:
                continue
            candidates.append(direction)

        # Collect this step's observations. observe_cell_in_direction does NOT
        # mutate `belief` — it just returns what was seen so we can compute the
        # update_trigger() against the PRIOR belief (the same state the planner
        # last optimised against) before applying changes.
        observations_this_step = []
        if candidates:
            ordered = lookaround_order_from_yaw(bot.yaw, candidates, bot)
            print(f"  [PLAN-LOOK] order: {ordered}")
            for direction in ordered:
                neighbor = physical_neighbor(current_physical_state, direction)
                obs = observe_cell_in_direction(neighbor, direction)
                if obs is not None:
                    observations_this_step.append(obs)

        # ─── Trigger: compare new perceived labels against the prior belief.
        # This is the same scheme as main.py — replan only when the
        # cell-by-cell mismatch between the prior probabilities and the
        # observed truth is non-trivial. Guarantees from the underlying
        # algorithm depend on this being the gating condition.
        just_observed = [o['cell'] for o in observations_this_step]
        trigger_function_value = 0.0
        if just_observed:
            neighbor_true_labels = {o['cell']: o['label'] for o in observations_this_step}
            previous_probabilities = {c: belief[c] for c in just_observed}
            previous_probabilities = {k: v.tolist() for k, v in previous_probabilities.items()}
            trigger_function_value = update_trigger(
                just_observed, neighbor_true_labels, previous_probabilities
            )
            print(f"  [TRIGGER] {trigger_function_value:.3f} (threshold={TRIGGER_THRESHOLD})")

        # Apply the deferred belief updates AFTER the trigger has been computed.
        # Hard update for the cell directly observed; soft update for any
        # inferred-far cell from the same observation.
        for obs in observations_this_step:
            belief = update(belief, obs['cell'], obs['label'])
            if obs['soft_target'] is not None and obs['soft_label'] is not None:
                belief = soft_update_belief(belief, obs['soft_target'], obs['soft_label'])
                print(f"  [SOFT] cell {obs['soft_target']} <- "
                      f"P({obs['soft_label']})={SOFT_P_LABEL}, P(empty)={SOFT_P_EMPTY}")
        observation_probabilities = belief

        # Replan only when the trigger fires — i.e., observations actually
        # disagreed with the prior. Otherwise reuse the existing policy.
        replan_needed = trigger_function_value > TRIGGER_THRESHOLD
        if replan_needed:
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
            print(f"  [REPLAN] trigger={trigger_function_value:.3f} > thr={TRIGGER_THRESHOLD}; "
                  f"new policy computed.")
        else:
            # No replan: reuse `policy` and `all_values` from the previous step.
            # adj_matrix already covers current_physical_state from the prior step.
            print(f"  [NO-REPLAN] trigger={trigger_function_value:.3f} <= thr={TRIGGER_THRESHOLD}; "
                  f"reusing existing policy.")

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

        # ─── Diagnostic: reconstruct Q(s, a) the same way Value_iteration does,
        # so [Q] matches what the policy actually argmaxes over. Earlier we
        # were printing V(successor) which is NOT what the planner argmaxes —
        # the planner uses Q = Σ_dfa-transition prob × (-1 + γ × V(next_PA_state)).
        # The previous diagnostic could disagree with the chosen action (e.g.
        # showed up=-12.58, down=-11.75, but policy picked up) because it
        # ignored the transition probability distribution and the per-step cost.
        GAMMA = gamma
        REWARD_STEP = -1.0   # matches a*c in Value_iteration with a=1, c=-1
        from planning import build_transition_map
        txn_map = build_transition_map(portion_transitions)

        def q_for_action(a):
            succ_phys = get_next_state(n, m, current_physical_state, a, adj_matrix)
            if succ_phys is None:
                return None, None
            # Look up all the (current_state, succ_phys) product-state transitions.
            possible = txn_map.get((current_state, succ_phys), [])
            if not possible:
                return None, succ_phys
            q = 0.0
            for j in possible:
                prob = transition_dict.get((repr(j[0]), repr(j[1])), 0.0)
                next_pa = j[1]
                if next_pa[1] == 'accept_all':
                    term = REWARD_STEP + 1.0     # b*r = 0*1 = 0 in the original; using a*c+b*r = -1+0
                elif next_pa[1] == 'Trash':
                    term = REWARD_STEP / (1 - GAMMA)
                else:
                    v_next = all_values.get(next_pa, 0.0)
                    term = REWARD_STEP + GAMMA * v_next
                q += prob * term
            return q, succ_phys

        legal_actions = ['up', 'down', 'left', 'right', 'stay']
        q_parts = []
        for a in legal_actions:
            q, succ_phys = q_for_action(a)
            if q is None:
                continue
            mark = '*' if a == action else ''
            q_parts.append(f"{a}->{succ_phys} Q={q:7.2f}{mark}")
        if q_parts:
            print(f"  [Q]      " + " | ".join(q_parts))
        print(f"  [V]      V(current)={current_value:.2f}")
        # Top belief mass per cell (only show cells with non-trivial mass).
        print(f"  [BELIEF]")
        for cell_idx in range(n * m):
            # Find top-2 labels by probability for this cell.
            top = sorted(belief[cell_idx], key=lambda pl: -pl[0])[:2]
            # Suppress cells whose top label is empty with mass > 0.95 (boring).
            if top[0][1] == EMPTY_LABEL and top[0][0] > 0.95:
                continue
            parts = ", ".join(f"P({lbl})={p:.2f}" for p, lbl in top)
            marker = f"  <- perceived: {perceived_labels[cell_idx]!r}" if cell_idx in perceived_labels else ""
            print(f"           cell {cell_idx:>2}: {parts}{marker}")

        print(f"  [DECIDE] action={action:<5} -> cell {next_physical_state} | value={current_value:8.2f}")

        # Refresh the live view with what we know at the current cell (post
        # look-around, pre-move): position, all perceived labels, and belief.
        viz.update(robot_cell=current_physical_state, heading=action,
                   perceived=perceived_labels, belief=belief,
                   dfa_state=str(current_dfa_state), step=step_count)

        if action == 'stay' or next_physical_state is None:
            # Policy picked 'stay'. Don't halt — force the best non-stay action
            # among the valid ones so the robot keeps exploring. We rank by the
            # value of the successor product-state (V(next)) and pick the highest;
            # ties broken by direction-name lexical order.
            candidates = []
            for a in ['up', 'down', 'left', 'right']:
                succ_phys = get_next_state(n, m, current_physical_state, a, adj_matrix)
                if succ_phys is None:
                    continue
                succ_label = perceived_labels.get(succ_phys, EMPTY_LABEL)
                succ_dfa = current_dfa_state
                for tr in dfa_transitions:
                    if tr[0] == current_dfa_state and succ_label == tr[1][0]:
                        succ_dfa = tr[2]
                        break
                v = all_values.get((succ_phys, succ_dfa), float('-inf'))
                candidates.append((v, a, succ_phys))
            if not candidates:
                print(f"  [WARN] no legal non-stay action at cell {current_physical_state}; halting.")
                break
            candidates.sort(key=lambda c: (-c[0], c[1]))
            v_forced, action, next_physical_state = candidates[0]
            print(f"  [FORCE]  policy said stay; forcing {action} -> cell {next_physical_state} "
                  f"(V(next)={v_forced:.2f}) to keep exploring.")

        # ─── Execute one cell move ──────────────────────────────────────
        # Settle dwell BEFORE the move: AMCL's particle filter can lag during
        # the in-place look-around rotations (LiDAR scans + odometry priors
        # don't constrain (x,y) well when the robot is spinning). If we start
        # the drive before AMCL re-converges, x0 = self.x is captured with a
        # stale pose, then self.x jumps catch-up *during* the drive, the PID
        # underestimates `traveled`, and the robot overshoots. A short dwell
        # gives AMCL a few translation-free scans to lock back in.
        time.sleep(0.5)

        pre_x, pre_y = bot.x, bot.y
        print(f"  [POSE-PRE]  x={pre_x:.3f} y={pre_y:.3f} yaw={bot.yaw:+.3f}")
        bot.move(action)
        bot.wait()
        post_x, post_y = bot.x, bot.y
        # Project the displacement onto the action's heading to get the real
        # signed forward distance moved. If it's much more than CELL_SIZE, we
        # overshot.
        import math as _math
        th = bot._action_target_yaw(action)
        moved = (post_x - pre_x) * _math.cos(th) + (post_y - pre_y) * _math.sin(th)
        print(f"  [POSE-POST] x={post_x:.3f} y={post_y:.3f} yaw={bot.yaw:+.3f}  "
              f"moved={moved*100:+.1f} cm (target {bot.CELL_SIZE*100:.0f} cm)")

        # Mark cell as visited.
        if next_physical_state not in visited_states_un:
            visited_states_un.append(next_physical_state)
        visited_states.append(current_physical_state)
        full_physical_traj.append(current_physical_state)

        # If we just entered a cell we never observed via look-around (rare —
        # planner-forced exploration into a non-candidate cell), treat the entry
        # itself as a direct observation that the cell is empty (we're standing
        # on it; if a marker were there we'd see it). Collapse the belief to
        # match.
        if next_physical_state not in perceived_labels:
            perceived_labels[next_physical_state] = EMPTY_LABEL
            belief = update(belief, next_physical_state, EMPTY_LABEL)
            observation_probabilities = belief
            print(f"  [ENTRY] cell {next_physical_state} not in look-around history; "
                  f"recording as empty on entry.")

        # Update DFA based on the cell just entered (whose label we observed
        # during the look-around at the previous step, or that we'll observe
        # next iteration if this is the first time we entered it).
        # label = perceived_labels.get(next_physical_state, EMPTY_LABEL)
        # for tr in dfa_transitions:
        #     if tr[0] == current_dfa_state and label == tr[1][0]:
        #         next_dfa_state = tr[2]
        label = canonical_label(perceived_labels.get(next_physical_state, EMPTY_LABEL))

        for tr in dfa_transitions:
            dfa_label = canonical_label(tr[1][0])

            if tr[0] == current_dfa_state and label == dfa_label:
                next_dfa_state = tr[2]
                break
        next_state = (next_physical_state, next_dfa_state)

        print(f"           entered cell {next_physical_state}; next_dfa={next_dfa_state}")

        # Refresh the live view at the newly-entered cell.
        viz.update(robot_cell=next_physical_state, heading=action,
                   perceived=perceived_labels, belief=belief,
                   dfa_state=str(next_dfa_state), step=step_count)

    bot.wait()
    viz.close()
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


# Tk must own the main thread (Tcl is not thread-safe); the planner runs in a
# worker thread. Calling _run_planner() directly with a threaded LiveViz caused
# "Tcl_AsyncDelete: async handler deleted by the wrong thread" at shutdown.
viz.run(_run_planner)