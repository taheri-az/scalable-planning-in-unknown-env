"""
analysis2.py — Monte-Carlo benchmark: belief generated FROM true map.

True map: β% of cells get non-empty labels (round-robin across a,b,c,d).

Three belief types (all get ζ-correction and renormalization):
  strong   – all β labeled cells: P(true_label) = 0.9
  moderate – β/2 labeled cells get P(true_label) = 0.9 (correct),
             β/2 random empty cells get P(random_region) = 0.9 (uninformative)
  random   – β random empty cells get P(random_region) = 0.9 (no link to true map)

Sweep: BETA_VALUES × BELIEF_TYPES × P_H_INIT_VALUES × REPLAN_FREQ_VALUES.
Maps are generated once per beta and shared across all other dimensions.

Usage:
    python analysis2.py
"""

import time
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from grid import create_graph
from labeling import get_states_within_h_distance, assign_probabilities_g3, update
from dfa import (
    extract_atomic_props,
    extract_dfa_transitions_with_trash_expanded,
    probabilistic_labeling_next,
)
from product_automaton import generate_product_automaton
from planning import (
    PA_values,
    Value_iteration,
    generate_and_visit,
    get_next_state,
    update_trigger,
    filter_adj_matrix,
    prune_dict_by_states,
    prune_transitions_by_states,
    find_paths_in_visited,
)


# ── Experiment configuration ───────────────────────────────────────────────────

N_RUNS_PER_CONFIG  = 100
BETA_VALUES        = [0.05]
BELIEF_TYPES       = ['strong', 'moderate', 'random']
P_H_INIT_VALUES    = [40,3]
REPLAN_FREQ_VALUES = [3]   # None = trigger-based, int = fixed every N steps
MAX_ATTEMPTS       = 300
GENERATE_MAPS      = False          # True = generate & save; False = load from maps.json

n, m        = 10, 10
h           = 2
THRESHOLD   = 0
GAMMA       = 0.99
EPSILON     = 0.01
FORMULA     = "F((a & F((b & F((c & F(d)))))))"
REGIONS     = ['a', 'b', 'c', 'd']
MAX_STEPS   = 500
RANDOM_SEED = 42

LOW_VALUE  = -1 / (1 - GAMMA) + 100 * EPSILON
OUTPUT_DIR = Path("analysis2_results")


# ── True map generation ────────────────────────────────────────────────────────

def generate_true_map(beta, n, m, regions, rng):
    """
    Assign non-empty labels to β·n·m cells using round-robin over regions.
    All other cells receive the empty label.
    Returns {state: label_string} for every cell.
    """
    e_label = ' && '.join(f'!{r}' for r in regions)
    region_labels = [
        ' && '.join(r if r == focus else f'!{r}' for r in regions)
        for focus in regions
    ]
    n_labeled    = max(len(regions), int(n * m * beta))
    labeled_cells = rng.sample(range(n * m), n_labeled)
    rng.shuffle(labeled_cells)

    true_labels = {s: e_label for s in range(n * m)}
    for i, s in enumerate(labeled_cells):
        true_labels[s] = region_labels[i % len(regions)]
    return true_labels


def is_valid_map(true_labels, regions):
    """Every region must appear as a positive literal in at least one cell."""
    for r in regions:
        if not any(
            r in [p.strip() for p in lbl.split('&&')]
            for lbl in true_labels.values()
        ):
            return False
    return True


def make_label_fn(true_labels):
    return lambda s: true_labels[s]


# ── Belief builders ────────────────────────────────────────────────────────────

def _region_labels(regions):
    return [
        ' && '.join(r if r == focus else f'!{r}' for r in regions)
        for focus in regions
    ]


def build_belief_strong(true_labels, regions, rng):
    """
    All β labeled cells: P(true_label) = 0.9.
    The remaining 0.1 is spread across other labels via ζ-correction.
    """
    e_label = ' && '.join(f'!{r}' for r in regions)
    return {
        s: {lbl: 0.9}
        for s, lbl in true_labels.items()
        if lbl != e_label
    }


def build_belief_moderate(true_labels, regions, rng):
    """
    β/2 labeled cells: P(true_label) = 0.9  (correct signal).
    β/2 random empty cells: P(random_region) = 0.9  (uninformative signal).
    """
    e_label  = ' && '.join(f'!{r}' for r in regions)
    r_labels = _region_labels(regions)

    labeled     = [s for s, lbl in true_labels.items() if lbl != e_label]
    empty_cells = [s for s, lbl in true_labels.items() if lbl == e_label]

    rng.shuffle(labeled)
    n_strong  = max(1, len(labeled) // 2)
    n_rand    = len(labeled) - n_strong
    rand_cells = rng.sample(empty_cells, min(n_rand, len(empty_cells)))

    initial_belief = {}
    for s in labeled[:n_strong]:
        initial_belief[s] = {true_labels[s]: 0.9}
    for s in rand_cells:
        initial_belief[s] = {rng.choice(r_labels): 0.9}
    return initial_belief


def build_belief_random(true_labels, regions, rng):
    """
    β random empty cells: P(random_region) = 0.9.
    No connection to the true map at all.
    """
    e_label  = ' && '.join(f'!{r}' for r in regions)
    r_labels = _region_labels(regions)

    labeled     = [s for s, lbl in true_labels.items() if lbl != e_label]
    empty_cells = [s for s, lbl in true_labels.items() if lbl == e_label]

    rand_cells = rng.sample(empty_cells, min(len(labeled), len(empty_cells)))
    return {s: {rng.choice(r_labels): 0.9} for s in rand_cells}


BELIEF_BUILDERS = {
    'strong':   build_belief_strong,
    'moderate': build_belief_moderate,
    'random':   build_belief_random,
}


# ── Map statistics ─────────────────────────────────────────────────────────────

def compute_map_stats(true_labels, regions, m):
    e_label = ' && '.join(f'!{r}' for r in regions)
    region_cells = {
        r: [s for s, lbl in true_labels.items()
            if r in [p.strip() for p in lbl.split('&&')]]
        for r in regions
    }
    counts = {f'n_cells_{r}': len(region_cells[r]) for r in regions}

    reps = []
    for r in regions:
        cells = region_cells[r]
        if cells:
            reps.append(min(cells, key=lambda s: s // m + s % m))

    if len(reps) >= 2:
        pairs = [
            abs(reps[i] // m - reps[j] // m) + abs(reps[i] % m - reps[j] % m)
            for i in range(len(reps))
            for j in range(i + 1, len(reps))
        ]
        mean_dist = round(sum(pairs) / len(pairs), 1)
    else:
        mean_dist = 0.0

    return counts, mean_dist


# ── Single-episode runner ──────────────────────────────────────────────────────

def run_episode(
    nodes, edges, adj_org,
    product_nodes, transitions,
    dfa_transitions, observations,
    atomic_props, initial_state_str,
    label_fn, initial_belief=None, p_h_init=3, replan_freq=None,
):
    t_start     = time.time()
    empty_label = ' && '.join(f'!{r}' for r in REGIONS)

    p_h        = p_h_init
    policy_p_h = p_h_init
    belief     = assign_probabilities_g3(n, m, atomic_props, initial_belief=initial_belief)
    obs_probs  = belief

    start_node     = (0, initial_state_str)
    current_state  = start_node
    next_state     = start_node
    next_dfa_state = initial_state_str
    current_phys   = 0

    plan_neighbors = get_states_within_h_distance(m, n, current_phys, p_h)
    adj_matrix     = filter_adj_matrix(adj_org, plan_neighbors)
    pruned_set     = prune_dict_by_states(PA_values(m, n, product_nodes, adj_matrix), plan_neighbors)
    portion_trans  = prune_transitions_by_states(transitions, plan_neighbors)
    trans_dict     = probabilistic_labeling_next(portion_trans, obs_probs, dfa_transitions, adj_matrix)
    policy, values = Value_iteration(
        m, n, pruned_set, trans_dict, portion_trans, product_nodes, GAMMA, adj_matrix, EPSILON
    )

    visited_states    = [0]
    visited_un        = [0]
    discovered_labels = []
    traj              = []

    for s in get_states_within_h_distance(m, n, current_phys, h):
        visited_un.append(s)

    current_value = values[current_state]
    p_t_t, p_t_c  = 0.0, 0
    max_p_h       = p_h
    p_h_history   = []
    j             = 0
    steps         = 0
    steps_hit_max = False

    while next_dfa_state != 'accept_all':
        if steps >= MAX_STEPS:
            steps_hit_max = True
            break
        steps += 1

        if not generate_and_visit(m, n, visited_un) and current_value < LOW_VALUE:
            break

        current_state = next_state
        current_dfa   = current_state[1]
        current_phys  = current_state[0]
        traj.append(current_phys)
        action        = policy[current_state]
        current_value = values[current_state]

        next_phys = get_next_state(m, n, current_phys, action, adj_matrix)
        if next_phys is None:
            break

        current_value_0 = values[current_state]
        h_neighbors     = get_states_within_h_distance(m, n, next_phys, h)
        plan_neighbors  = get_states_within_h_distance(m, n, next_phys, p_h)
        adj_matrix      = filter_adj_matrix(adj_org, plan_neighbors)

        if current_value_0 > LOW_VALUE:
            p_h = p_h_init

        while current_value_0 < LOW_VALUE:
            p_h    += 1
            max_p_h = max(max_p_h, p_h)
            plan_neighbors = get_states_within_h_distance(m, n, next_phys, p_h)
            _, new_states  = find_paths_in_visited(n, m, next_phys, discovered_labels)
            for s in new_states:
                if s not in plan_neighbors:
                    plan_neighbors.append(s)
            adj_matrix    = filter_adj_matrix(adj_org, plan_neighbors)
            pruned_set    = prune_dict_by_states(PA_values(m, n, product_nodes, adj_matrix), plan_neighbors)
            policy_p_h    = p_h
            portion_trans = prune_transitions_by_states(transitions, plan_neighbors)
            trans_dict    = probabilistic_labeling_next(portion_trans, obs_probs, dfa_transitions, adj_matrix)
            t0 = time.time()
            policy, values = Value_iteration(
                m, n, pruned_set, trans_dict, portion_trans, product_nodes, GAMMA, adj_matrix, EPSILON
            )
            p_t_t += time.time() - t0
            p_t_c += 1
            current_value_0 = values[current_state]

        p_h_history.append(p_h)

        for s in h_neighbors:
            if s not in visited_un:
                visited_un.append(s)
        visited_states.append(current_phys)

        neighbor_labels = {s: label_fn(s) for s in h_neighbors}
        prev_probs      = {s: list(belief[s]) for s in h_neighbors}
        trigger_val     = update_trigger(h_neighbors, neighbor_labels, prev_probs)

        for s in h_neighbors:
            nl = label_fn(s)
            belief = update(belief, s, nl)
            if nl != empty_label and s not in discovered_labels:
                discovered_labels.append(s)

        label = label_fn(next_phys)
        for tr in dfa_transitions:
            if tr[0] == current_dfa and label == tr[1][0]:
                next_dfa_state = tr[2]

        next_state  = (next_phys, next_dfa_state)
        next_value  = values[next_state]
        belief      = update(belief, next_phys, label)
        obs_probs   = belief

        j += 1
        if replan_freq is None:
            should_replan = trigger_val > THRESHOLD or j >= policy_p_h - 1
        else:
            should_replan = j >= replan_freq
        if should_replan:
            j = 0
            _, new_states  = find_paths_in_visited(n, m, next_phys, discovered_labels)
            for s in new_states:
                if s not in plan_neighbors:
                    plan_neighbors.append(s)
            adj_matrix    = filter_adj_matrix(adj_org, plan_neighbors)
            pruned_set    = prune_dict_by_states(PA_values(m, n, product_nodes, adj_matrix), plan_neighbors)
            policy_p_h    = p_h
            portion_trans = prune_transitions_by_states(transitions, plan_neighbors)
            trans_dict    = probabilistic_labeling_next(portion_trans, obs_probs, dfa_transitions, adj_matrix)
            t0 = time.time()
            policy, values = Value_iteration(
                m, n, pruned_set, trans_dict, portion_trans, product_nodes, GAMMA, adj_matrix, EPSILON
            )
            p_t_t += time.time() - t0
            p_t_c += 1
            if current_phys not in visited_un:
                visited_un.append(current_phys)

    completed = next_dfa_state == 'accept_all'
    traj.append(next_phys)

    return {
        'completed':             completed,
        'trajectory_length':     len(traj),
        'replanning_count':      p_t_c,
        'total_replan_time_s':   round(p_t_t, 4),
        'avg_replanning_time_s': round(p_t_t / p_t_c, 4) if p_t_c > 0 else 0.0,
        'total_time_s':          round(time.time() - t_start, 2),
        'max_p_h':               max_p_h,
        'avg_p_h':               round(sum(p_h_history) / len(p_h_history), 2) if p_h_history else float(p_h_init),
        'steps_hit_max':         steps_hit_max,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Building DFA and product automaton (shared across all runs)...")
    atomic_props = extract_atomic_props(FORMULA)
    dfa_transitions, initial_state, _ = extract_dfa_transitions_with_trash_expanded(FORMULA)
    dfa_states        = list({t[0] for t in dfa_transitions} | {t[2] for t in dfa_transitions})
    observations      = list({cond for _, conds, _ in dfa_transitions for cond in conds})
    initial_state_str = str(initial_state)

    nodes, edges, adj_np = create_graph(n, m)
    adj_org = adj_np.tolist()

    t0 = time.time()
    _, transitions, product_nodes, _ = generate_product_automaton(
        nodes, edges, adj_org, dfa_states, dfa_transitions, observations
    )
    transitions = list(dict.fromkeys(transitions))
    print(f"Product automaton: {len(product_nodes)} states, built in {time.time()-t0:.1f}s\n")

    rng      = random.Random(RANDOM_SEED)
    results  = []
    maps_path = OUTPUT_DIR / 'maps_new_0.05.json'

    # ── Load or generate true maps ─────────────────────────────────────────────
    if GENERATE_MAPS:
        all_maps = {}
        for beta in BETA_VALUES:
            cfg_key      = f'beta{int(beta * 100):03d}'
            sampled_maps = []
            attempts     = 0
            while len(sampled_maps) < N_RUNS_PER_CONFIG:
                if attempts >= MAX_ATTEMPTS:
                    print(f"  [β={int(beta*100)}%] gave up after {attempts} attempts "
                          f"({len(sampled_maps)}/{N_RUNS_PER_CONFIG} maps sampled)")
                    break
                attempts += 1
                true_labels = generate_true_map(beta, n, m, REGIONS, rng)
                if is_valid_map(true_labels, REGIONS):
                    sampled_maps.append(true_labels)
            all_maps[cfg_key] = [{str(k): v for k, v in mp.items()} for mp in sampled_maps]

        with open(maps_path, 'w') as f:
            json.dump(all_maps, f, indent=2)
        print(f"Generated and saved {sum(len(v) for v in all_maps.values())} maps → {maps_path}\n")
    else:
        if not maps_path.exists():
            raise FileNotFoundError(f"GENERATE_MAPS=False but {maps_path} does not exist.")
        with open(maps_path) as f:
            all_maps = json.load(f)
        print(f"Loaded maps from {maps_path} "
              f"({sum(len(v) for v in all_maps.values())} maps across {len(all_maps)} configs)\n")

    # ── Run episodes ───────────────────────────────────────────────────────────
    total_runs = (len(BETA_VALUES) * N_RUNS_PER_CONFIG *
                  len(BELIEF_TYPES) * len(P_H_INIT_VALUES) * len(REPLAN_FREQ_VALUES))
    pbar = tqdm(total=total_runs, unit='run', dynamic_ncols=True)

    for beta in BETA_VALUES:
        cfg_key      = f'beta{int(beta * 100):03d}'
        sampled_maps = [{int(k): v for k, v in mp.items()} for mp in all_maps.get(cfg_key, [])]

        for belief_type in BELIEF_TYPES:
            builder = BELIEF_BUILDERS[belief_type]

            for p_h_init in P_H_INIT_VALUES:
                for replan_freq in REPLAN_FREQ_VALUES:
                    freq_label = 'trigger' if replan_freq is None else f'every{replan_freq}'

                    for run_idx, true_labels in enumerate(sampled_maps):
                        label_fn          = make_label_fn(true_labels)
                        counts, mean_dist = compute_map_stats(true_labels, REGIONS, m)
                        initial_belief    = builder(true_labels, REGIONS, rng)

                        pbar.set_description(
                            f'β={int(beta*100):3d}% {belief_type:<8} p_h={p_h_init} rf={freq_label}'
                        )

                        try:
                            stats = run_episode(
                                nodes, edges, adj_org,
                                product_nodes, transitions,
                                dfa_transitions, observations,
                                atomic_props, initial_state_str,
                                label_fn, initial_belief=initial_belief,
                                p_h_init=p_h_init, replan_freq=replan_freq,
                            )
                        except Exception as exc:
                            tqdm.write(f"  [{cfg_key} {belief_type} p_h={p_h_init} "
                                       f"rf={freq_label} run={run_idx+1}] ERROR: {exc}")
                            stats = {
                                'completed': False, 'trajectory_length': 0,
                                'replanning_count': 0, 'total_replan_time_s': 0.0,
                                'avg_replanning_time_s': 0.0, 'total_time_s': 0.0,
                                'max_p_h': p_h_init, 'avg_p_h': float(p_h_init),
                                'steps_hit_max': False,
                            }

                        row = {
                            'beta':                  beta,
                            'belief_type':           belief_type,
                            'p_h_init':              p_h_init,
                            'replan_freq':           freq_label,
                            'run':                   run_idx + 1,
                            'completed':             stats['completed'],
                            'trajectory_length':     stats['trajectory_length'],
                            'replanning_count':      stats['replanning_count'],
                            'avg_replanning_time_s': stats['avg_replanning_time_s'],
                            'total_replan_time_s':   stats['total_replan_time_s'],
                            'total_time_s':          stats['total_time_s'],
                            'avg_p_h':               stats['avg_p_h'],
                            'max_p_h':               stats['max_p_h'],
                            'steps_hit_max':         stats['steps_hit_max'],
                            'mean_region_distance':  mean_dist,
                            **counts,
                        }
                        results.append(row)

                        status = '✓' if stats['completed'] else ('T' if stats['steps_hit_max'] else '✗')
                        pbar.set_postfix({
                            'status': status,
                            'len':    stats['trajectory_length'],
                            'replan': stats['replanning_count'],
                            'time':   f"{stats['total_time_s']:.1f}s",
                        })
                        tqdm.write(
                            f"[β={int(beta*100):3d}% {belief_type:<8} p_h={p_h_init} "
                            f"rf={freq_label} {run_idx+1:2d}/{N_RUNS_PER_CONFIG}] {status} | "
                            f"len={stats['trajectory_length']:5d} | "
                            f"replan={stats['replanning_count']:4d} | "
                            f"avg_t={stats['avg_replanning_time_s']:.3f}s | "
                            f"total={stats['total_time_s']:6.1f}s | "
                            f"dist={mean_dist:.1f}"
                        )
                        pbar.update(1)

    pbar.close()

    # ── Save results ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'results.csv', index=False)
    print(f"\nSaved per-run results → {OUTPUT_DIR / 'results.csv'}")

    # ── Print summary ──────────────────────────────────────────────────────────

    def fmt(s):
        return f"mean={s.mean():.1f}  std={s.std():.1f}  min={s.min():.0f}  max={s.max():.0f}"

    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY")
    print("=" * 85)
    print(f"  Beta fractions       : {BETA_VALUES}")
    print(f"  Belief types         : {BELIEF_TYPES}")
    print(f"  P_H_INIT values      : {P_H_INIT_VALUES}")
    print(f"  Replan freq values   : {REPLAN_FREQ_VALUES}  (None = trigger-based)")
    print(f"  Total runs           : {len(df)}")
    print(f"  Completed (✓)        : {df['completed'].sum()}  ({df['completed'].mean()*100:.1f}%)")
    print(f"  Timeout  (T)         : {df['steps_hit_max'].sum()}")
    print(f"  Failed   (✗)         : {(~df['completed'] & ~df['steps_hit_max']).sum()}")

    # Success rate: belief_type × beta (averaged over p_h and replan_freq)
    print("\n  Success rate (%) — belief_type × beta:")
    pivot = (
        df.groupby(['belief_type', 'beta'])['completed']
        .mean().mul(100).round(1).unstack(level='beta')
    )
    pivot.columns = [f"β={int(b*100)}%" for b in pivot.columns]
    print(pivot.to_string())

    # Per-config stats (completed runs only)
    print("\n  Per-config stats (completed runs only):")
    header = (f"  {'config':<38} {'n':>4}  {'traj':>8}  {'replan':>8}  "
              f"{'replan_t(s)':>12}  {'avg_p_h':>8}  {'time(s)':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (beta, belief_type, p_h_init, replan_freq), grp in df.groupby(
            ['beta', 'belief_type', 'p_h_init', 'replan_freq']):
        done  = grp[grp['completed']]
        label = f"β={int(beta*100):3d}% {belief_type:<8} p_h={p_h_init} rf={replan_freq}"
        if len(done):
            print(f"  {label:<38} {len(done):>4}  "
                  f"{done['trajectory_length'].mean():>8.1f}  "
                  f"{done['replanning_count'].mean():>8.1f}  "
                  f"{done['avg_replanning_time_s'].mean():>12.3f}  "
                  f"{done['avg_p_h'].mean():>8.2f}  "
                  f"{done['total_time_s'].mean():>8.1f}")
        else:
            print(f"  {label:<38} {0:>4}  (no completions)")

    print("=" * 85)
