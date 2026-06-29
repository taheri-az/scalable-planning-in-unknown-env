"""
Interactive belief builder for the look-around planner.

Run this BEFORE main_lookaround.py to construct the initial belief grid by
hand instead of hard-coding `initial_belief` in the planner source.

Flow:
  1. Ask for grid size (rows n, cols m).
  2. Ask for the atomic propositions (e.g. a b c).
  3. Show the grid; let you pick a cell and assign probabilities to one or
     more atomic-proposition labels for that cell. Repeat until done.
  4. Apply the zeta floor + renormalize via assign_probabilities_g3 (the same
     routine the planner uses) and pickle the result.

The saved pickle is a numpy object array of shape (n*m, 2^len(atomics)) where
each entry is (probability, label_string) — exactly what
assign_probabilities_g3 returns, so main_lookaround.py can load it directly.

Cell numbering matches the planner: cell = row * m + col  (stride = m).

Usage:
    python3 build_belief.py                 # interactive, saves belief.pkl
    python3 build_belief.py --out my.pkl    # custom output path
"""

import sys
import pickle
import argparse
import itertools

import numpy as np

from labeling import assign_probabilities_g3


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return str(default)
        if raw != "":
            return raw


def ask_int(prompt, default=None, lo=None, hi=None):
    while True:
        raw = ask(prompt, default)
        try:
            v = int(raw)
        except ValueError:
            print("  ! please enter a whole number.")
            continue
        if lo is not None and v < lo:
            print(f"  ! must be >= {lo}.")
            continue
        if hi is not None and v > hi:
            print(f"  ! must be <= {hi}.")
            continue
        return v


def ask_float(prompt, lo=0.0, hi=1.0):
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            v = float(raw)
        except ValueError:
            print("  ! please enter a number.")
            continue
        if not (lo <= v <= hi):
            print(f"  ! must be between {lo} and {hi}.")
            continue
        return v


def all_labels(atomics):
    """All 2^k conjunctions over the atomics, in assign_probabilities_g3's
    canonical order (itertools.product of [0,1] per atom)."""
    return [
        ' && '.join(atomics[i] if bits[i] else f'!{atomics[i]}'
                    for i in range(len(atomics)))
        for bits in itertools.product([0, 1], repeat=len(atomics))
    ]


def empty_label(atomics):
    return ' && '.join(f'!{a}' for a in atomics)


def print_grid(n, m, initial_belief, atomics):
    """Render the grid with cell indices; mark cells that already have a
    user-assigned prior with a '*'."""
    print()
    print("  Grid (cell = row*m + col):")
    width = max(4, len(str(n * m - 1)) + 2)
    for r in range(n):
        cells = []
        for c in range(m):
            cell = r * m + c
            mark = "*" if cell in initial_belief else " "
            cells.append(f"{cell:>{width-1}}{mark}")
        print("   " + "".join(cells))
    if initial_belief:
        print("   ( * = has a custom prior )")
    print()


def summarize_assignment(cell, dist, atomics):
    parts = ", ".join(f"P({lbl})={p:.2f}" for lbl, p in dist.items())
    print(f"   -> cell {cell}: {parts}")


def assign_cell(cell, atomics):
    """Interactively assign probabilities to labels for one cell.

    The user names labels by listing which atomics are TRUE (e.g. 'a' means
    'a && !b && !c'; 'a b' means 'a && b && !c'; 'empty' or '' means the
    all-negated label). The remaining mass after the entered labels is left
    for assign_probabilities_g3 to spread via the zeta floor — but if the
    user's entered probabilities don't sum to 1 we explicitly put the
    remainder on the empty label so the prior is well-defined.
    """
    labels = all_labels(atomics)
    emp = empty_label(atomics)
    print(f"\n  Assigning cell {cell}.")
    print(f"    Atomics: {', '.join(atomics)}")
    print( "    For each label, list the TRUE atomics separated by spaces.")
    print( "    Examples:  'a'  -> "
           f"{labels[0] if False else '(' + atomics[0] + ' true, rest false)'}")
    print( "               'a b'  -> both true, rest false")
    print( "               'empty' or blank  -> all false")
    print( "    Enter 'done' when finished with this cell.")

    dist = {}
    while True:
        raw = input(f"    cell {cell} label (true atomics / 'done'): ").strip()
        if raw.lower() in ("done", "d"):
            break
        if raw.lower() in ("empty", "none", "e"):
            chosen = emp
        elif raw == "":
            chosen = emp
        else:
            toks = raw.split()
            bad = [t for t in toks if t not in atomics]
            if bad:
                print(f"      ! unknown atomics {bad}; valid: {atomics}")
                continue
            chosen = ' && '.join(a if a in toks else f'!{a}' for a in atomics)
        p = ask_float(f"      P({chosen})")
        dist[chosen] = p
        running = sum(dist.values())
        print(f"      (running total assigned at this cell: {running:.2f})")
        if running > 1.0 + 1e-9:
            print("      ! total exceeds 1.0 — adjust before finishing.")

    if not dist:
        print("    (no labels entered; cell left at default prior)")
        return None

    total = sum(dist.values())
    if total > 1.0 + 1e-9:
        print(f"    ! total {total:.2f} > 1.0; please redo this cell.")
        return assign_cell(cell, atomics)

    # Put any remaining mass explicitly on the empty label so the prior is
    # complete (assign_probabilities_g3 still applies zeta to the zeros).
    if total < 1.0 - 1e-9 and emp not in dist:
        dist[emp] = round(1.0 - total, 6)
        print(f"    (remaining {dist[emp]:.2f} mass placed on empty label)")

    summarize_assignment(cell, dist, atomics)
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="belief.pkl",
                    help="output pickle path (default: belief.pkl)")
    ap.add_argument("--zeta", type=float, default=0.02,
                    help="zeta floor for zero-probability labels (default 0.02)")
    args = ap.parse_args()

    print("=" * 60)
    print("Interactive belief builder")
    print("=" * 60)

    n = ask_int("Number of rows (n)", default=6, lo=1)
    m = ask_int("Number of columns (m)", default=3, lo=1)

    raw_atoms = ask("Atomic propositions (space-separated, e.g. a b c)",
                    default="a b c")
    atomics = raw_atoms.split()
    if not atomics:
        print("No atomics given; aborting.")
        sys.exit(1)
    if len(set(atomics)) != len(atomics):
        print("Duplicate atomics; aborting.")
        sys.exit(1)
    print(f"  Atomics: {atomics}  ->  {2**len(atomics)} possible labels/cell")

    initial_belief = {}
    print_grid(n, m, initial_belief, atomics)

    while True:
        raw = input("Pick a cell to assign (index), or 'done' to finish: ").strip()
        if raw.lower() in ("done", "d", ""):
            break
        try:
            cell = int(raw)
        except ValueError:
            print("  ! enter a cell index or 'done'.")
            continue
        if not (0 <= cell < n * m):
            print(f"  ! cell must be in 0..{n*m - 1}.")
            continue
        if cell in initial_belief:
            ow = ask(f"  cell {cell} already assigned; overwrite? (y/n)", "n")
            if ow.lower() not in ("y", "yes"):
                continue
        dist = assign_cell(cell, atomics)
        if dist is not None:
            initial_belief[cell] = dist
        print_grid(n, m, initial_belief, atomics)

    if not initial_belief:
        print("\nNo custom priors assigned. All cells will get the "
              "empty-biased default. Continuing anyway.")

    # Apply the zeta floor + renormalize via the SAME routine the planner uses.
    belief = assign_probabilities_g3(
        n, m, atomics, initial_belief=initial_belief, zeta=args.zeta
    )

    print("\n" + "=" * 60)
    print("Final belief (top-2 labels per cell with custom priors):")
    print("=" * 60)
    for cell in range(n * m):
        top = sorted(belief[cell], key=lambda pl: -pl[0])[:2]
        emp = empty_label(atomics)
        if cell not in initial_belief and top[0][1] == emp:
            continue
        parts = ", ".join(f"P({lbl})={p:.3f}" for p, lbl in top)
        print(f"  cell {cell:>2}: {parts}")

    payload = {
        "n": n,
        "m": m,
        "atomics": atomics,
        "zeta": args.zeta,
        "initial_belief": initial_belief,
        "belief": belief,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved belief to {args.out}")
    print("Load it in main_lookaround.py with:")
    print(f"    import pickle")
    print(f"    with open('{args.out}', 'rb') as f: data = pickle.load(f)")
    print(f"    belief = data['belief']   # n={n} m={m} atomics={atomics}")


if __name__ == "__main__":
    main()
