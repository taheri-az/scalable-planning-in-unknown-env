# Project Context — Scalable Planning in Unknown Env (TurtleBot3 + LTL)

## What this project is

A real-world implementation of LTL-guided planning on a TurtleBot3 Burger. The robot navigates a discrete grid, observes cell labels via a USB camera (Logitech C270), and follows a temporal-logic specification compiled to a DFA. Two main loop variants: `main.py` (observe-and-move, one cell at a time) and `main_lookaround.py` (rotate to observe each unmapped neighbor before each move).

## Hardware / software

- **TurtleBot3 Burger** running ROS Noetic on a Raspberry Pi (Ubuntu 20.04 ARM).
- **OpenCR** for motor control via rosserial.
- **LD08 / LDS-02 LiDAR** — driver: `ld08_driver` (NOT the older `hls_lfcd_lds_driver`).
- **Logitech C270** USB camera, plugged into the Pi. Camera focal length ≈ 1030 px after recalibration with 10×10 cm sheets at 50 cm.
- **Laptop**: Ubuntu, runs roscore (when in Pi-master mode it's reversed — see ROS network), rviz, optionally AMCL + map_server.
- **Python env**: `tl-planning` conda env on the Pi (has `spot`, `buddy`, `opencv`, etc.).

## Current network configuration (DHCP — re-verify if stale)

- **Pi (TurtleBot)**: `192.168.0.145`
- **Laptop (PC)**: `192.168.0.114`

Both `.bashrc` exports point ROS master at the Pi:
```
export ROS_MASTER_URI=http://192.168.0.145:11311
export ROS_HOSTNAME=<own_ip>
```

These IPs change each network. The user is at GitHub `taheri-az`.

## Grid + task

- `(n, m) = (6, 3)` — 6 rows × 3 columns = 18 cells. Stride is `m=3`. **Important**: `planning.py` and `labeling.py` internally use the second arg as the column-stride. `main.py` calls them with `(n, m, ...)` swapped from what it might naively look like — this was a real bug that only manifested on non-square grids. See commit `5fefd2e`.
- Cell numbering: `cell = row * 3 + col`. Cell 0 = (row 0, col 0). Cell 9 = (row 3, col 0).
- `CELL_SIZE_M = 0.5` (both `main.py` and `turtle_driver.py` must match).
- `ASSIGN_DIST_M = 0.65` — markers detected within this distance get hard-assigned; beyond → soft hint at a further cell.
- `formula_str = "F((a & F((b & F(c)))))"` — visit red, then yellow, then green in order. DFA has no Trash state for this formula — out-of-order observations just loop back.
- `p_h = 4` — planning horizon.
- Initial belief seeded at cell 3 (red, a), cell 4 (yellow, b), cell 9 (green, c) each at P=0.8 with P(empty)=0.2 — but **the physical marker layout changes between runs**, ask the user.

## Color → label mapping (label_detector.py)

```python
COLOR_LABEL = {
    "red":    "a && !b && !c",
    "yellow": "!a && b && !c",
    "green":  "!a && !b && c",
    "black":  None,
    "blue":   None,
    "orange": None,
    "purple": None,   # was added/removed during experimentation
}
```

The user iterated through several alternatives (black-for-b, purple-for-c) when lighting issues made yellow/green ambiguous, then went back to red/yellow/green with 10×10 cm color sheets. `Camera_tu/calibration.json` has `real_width: 10.0, focal_length_px: 1030.0`.

## Key files

- `main.py` — the original observe-then-move loop (DFA-aware planner with `update_trigger` gating replan).
- `main_lookaround.py` — the variant where the robot rotates at each cell to observe every unmapped 1-hop neighbor before moving. Uses `bot.face(direction)` to pivot without driving. Trigger-gated replan, same algorithm guarantees.
- `turtle_driver.py` — motion controller. TurtleBot class with PID (PD on yaw, P on forward). Reads AMCL pose via tf2_ros (or falls back to `/odom`). Has `move(action)`, `face(direction)`, `wait()`, `wait_for_cell_entry()`, `wait_for_rotation_done()`.
- `label_detector.py` — OpenCV HSV + pinhole distance estimator. Background grab thread keeps `_min_dist_per_color` and `_clipped_per_color`. `reset_observation_window()` clears them. `detect()` returns the closest color seen.
- `Camera_tu/distance_estimation.py` — interactive HSV tuner. `--color X --camera 0 --width 10` after re-calibration with 10 cm sheets. Press SPACE during initial calibration phase, `s` to save HSV, `q` to quit.
- `Camera_tu/hsv_bounds.json` — current HSV bounds, edited by the tuner. Always verify in-situ.
- `Camera_tu/calibration.json` — `{focal_length_px, real_width}`. Real width = 10 cm now.
- `labeling.py` — belief math. `update(grid, state, label)` collapses to a Dirac at observed label. `assign_probabilities_g3` is the seeded-prior initializer.
- `planning.py` — Value_iteration over the product automaton. Reward `c=-1` per step, `a*c+b*r = -1` for accept, `c/(1-γ) = -100` for Trash. Returns `(policy, all_values)` — `all_values[state]` is V*(s), not per-action Q.
- `dfa.py` — uses `spot` to compile LTL to DFA, expanding transitions over all valuations.
- `product_automaton.py` — builds the product MDP.

## Recurring bugs we fixed (do NOT reintroduce)

1. **`n` vs `m` swap on non-square grids** (`5fefd2e`). main.py was passing `(m, n, ...)` to planning/labeling functions whose internal stride is the second arg. Fixed by passing `(n, m, ...)` consistently. Only visible when n ≠ m.
2. **`MOTION_TIMEOUT = 12 s` was capping cell traversal** (`9bb044d`). At `LINEAR_SPEED = 0.035 m/s`, 12 s caps at 0.42 m regardless of `CELL_SIZE`. Bumped to 40 s. Bumping cell size had no effect because the timeout fired first.
3. **`face()` not invalidating `_cell_start_xy`** (`7844e93`). After `face()`, a subsequent `move()` in the same direction took the `same_direction` branch and computed `x0` from the old `_cell_start_xy + CELL_SIZE` — placing the start anchor one cell past where the robot actually was. Robot overshot by up to a cell. Now `face()` sets `_cell_start_xy = None`.
4. **Detector window reset BEFORE rotation in look-around** caused mid-rotation frames to latch bogus close-range distances. Reset moved to AFTER `face()` completes.
5. **`update_trigger` removed from main_lookaround**, breaking algorithm guarantees. Restored in `99768dd` — observe → compute trigger over PRIOR belief → apply updates → replan only if trigger > threshold. **Do not remove this gating.**
6. **Belief not collapsed on EMPTY observation** (`6e66650`). When a cell was looked at and seen empty, `perceived_labels[cell] = EMPTY` was set but `belief[cell]` stayed at the seeded prior — making all Q-values tie and the planner pick `stay`. Now `update(belief, cell, EMPTY_LABEL)` runs for both empty and non-empty observations.
7. **Belief not collapsed at starting cell or on entry** (`d459c04`). `perceived_labels = {0: EMPTY}` was set but `belief[0]` stayed at the prior. Now collapse at startup; also if a cell is entered without a prior look-around observation, mark it empty + collapse.
8. **AMCL pose lag after look-around rotations** (`f267ffb`). LiDAR + odom prior doesn't constrain (x,y) well during in-place rotations. After the look-around, AMCL takes a moment to re-converge. Added 0.5 s `time.sleep()` before each `bot.move()` so `x0 = self.x` captures a stable pose. Also added `[POSE-PRE]` / `[POSE-POST]` diagnostics showing the projected forward distance moved per cell.
9. **`[SUCC]` diagnostic was misleading** (`717ef87`). Old version printed V(successor) per action, but policy is argmax Q, not argmax V(succ). Replaced with `[Q]` that reconstructs Q the same way Value_iteration does: `Q = Σ prob × (-1 + γ × V(next_PA_state))`.

## Behaviors and decisions to remember

- **Sticky labels**: once a cell has a non-empty label in `perceived_labels`, never overwrite it. World is static.
- **Look-around order is heading-aware** (`6ee6149`): greedy nearest-yaw-first to minimize total rotation. Optimal for 4 cardinal directions.
- **Skip already-observed cells in look-around** regardless of label (empty or non-empty). No re-observation needed.
- **Force-non-stay** if policy picks `stay`: pick the legal action with the highest V(succ) and proceed. Otherwise the robot stalls when Q values tie.
- **AMCL on**: `bot = TurtleBot()` (use_amcl=True is the default). Brief experiment with `use_amcl=False` (odom-only) showed too much drift; reverted.
- **Map**: latest is `~/grid3_map.yaml` (and `~/grid4_map.yaml` if re-mapped). Always edit the yaml's `image:` line to be relative (`image: grid3_map.pgm`) so it works on either machine after SCP.

## Launch sequence (Pi-as-master, AMCL on Pi)

The user wants to keep the Pi as the master with everything on the Pi for now. Old "laptop runs AMCL" path was abandoned due to firewall / multi-master headaches.

**Terminal A** (Pi) — bringup:
```bash
ssh ubuntu@192.168.0.145
conda deactivate
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.145:11311
export ROS_HOSTNAME=192.168.0.145
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```

**Terminal B** (Pi) — navigation (ignores rviz + dwa_local_planner errors):
```bash
ssh ubuntu@192.168.0.145
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.145:11311
export ROS_HOSTNAME=192.168.0.145
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/grid3_map.yaml
```

**Terminal C** (Pi) — AMCL tuning + initial pose:
```bash
ssh ubuntu@192.168.0.145
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.145:11311
export ROS_HOSTNAME=192.168.0.145

rosparam set /amcl/update_min_d 0.05
rosparam set /amcl/update_min_a 0.05
rosparam set /amcl/min_particles 1000
rosparam set /amcl/max_particles 5000
rosnode kill /amcl
sleep 3
# physically place the robot at cell-0 corner facing +x:
rostopic pub /initialpose geometry_msgs/PoseWithCovarianceStamped \
  "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.07]}}" --once
rosrun tf tf_echo map base_footprint
```

**Terminal D** (Pi) — planner:
```bash
ssh ubuntu@192.168.0.145
conda activate tl-planning
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.145:11311
export ROS_HOSTNAME=192.168.0.145
cd ~/scalable-planning-in-unknown-env
git pull
python3 main.py            # or main_lookaround.py
```

Look for `TurtleBot: using AMCL pose (map -> base_footprint)` confirming AMCL works.

## Git workflow

- The Pi cannot push to GitHub (no SSH key set up). Workflow: the user pastes file contents from the Pi (e.g. `cat hsv_bounds.json`), the assistant writes the same content to the laptop's working copy and pushes from there.
- The Pi pulls. If `git pull` complains about local changes (typically because in-situ tuning modified `hsv_bounds.json`), run `git checkout -- <file>` to discard the local change before pulling.
- Commits use the assistant's standard format with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

## Diagnostics produced by the planners

`main_lookaround.py` prints, per step:
- `[Step N] at cell X | dfa=Y`
- `[PLAN-LOOK] order: [dir1, dir2, ...]` — heading-optimal look-around order
- `[LOOK] facing <dir> -> cell K` + `[DETECT] color=Xcm, ...` + `[LABEL] cell K -> label` or `cell K empty + soft hint cell L`
- `[TRIGGER] value (threshold=0.0)`
- `[REPLAN]` or `[NO-REPLAN]`
- `[Q] up->X Q=..., down->Y Q=..., ...` (the action with `*` is what policy picked)
- `[V] V(current)=...`
- `[BELIEF]` — per-cell top-2 labels (cells whose top is empty with P>0.95 are suppressed)
- `[DECIDE] action=... -> cell ... | value=...`
- `[POSE-PRE]`, `[POSE-POST]` showing AMCL position before/after the move and the projected forward distance moved
- `[FORCE]` if policy said stay and we forced the best non-stay action
- `[ENTRY]` if the destination cell wasn't in the look-around history

## User preferences and style

- The user wants concise, complete answers. They flag when summaries get too long.
- They prefer to test on hardware and iterate; quick experimental changes are welcome.
- They expect me to push commits from the laptop when they paste Pi-side file contents.
- They get frustrated by removing parts of their algorithm without asking (e.g. removing `update_trigger`). The algorithm has formal guarantees that depend on specific structures; don't simplify those away.
- They prefer me to flag risky changes up front (e.g. HSV bounds that look prone to cross-detection) rather than just commit them silently.
- Video records to `run.mp4` (main.py) or `run_lookaround.mp4` (lookaround variant). Files get timestamped by the user before re-running. Download with:
  ```
  scp ubuntu@192.168.0.145:~/scalable-planning-in-unknown-env/run.mp4 ~/run_$(date +%Y%m%d_%H%M%S).mp4
  ```

## Recent commit history (top → bottom = newest → older)

```
d459c04 Collapse belief at starting cell and on entry-without-prior-observation
717ef87 Diagnostic: reconstruct true Q(s,a) so [Q] matches policy's argmax
f267ffb Pre-move AMCL settle dwell + pose diagnostics to measure overshoot
7844e93 face(): invalidate _cell_start_xy so subsequent move() doesn't overshoot
99768dd Restore update_trigger gating: replan only when trigger > threshold
6e66650 Collapse cell belief on EMPTY observation too (was leaving stale prior)
6ee6149 Heading-aware look-around order: visit closest direction first
fcb8a74 Look-around: reset detector AFTER face() finishes; skip already-observed cells; show V(succ) per action
ca1ffdb Look-around: print Q-values + belief at each step; force best non-stay action
81a8c37 Look-around: wait for full cell move to finish before next iteration
60b8b59 Add look-around variant: rotate to observe each neighbor before moving
46cd45c Bump LINEAR_SPEED 0.035 -> 0.05 m/s
3290087 Re-enable AMCL for new (grid4) map
a613901 Cell size back to 60 cm now that MOTION_TIMEOUT is no longer the cap
9bb044d Bump MOTION_TIMEOUT 12 s -> 40 s (was capping physical distance per cell)
8e31f94 Re-enable AMCL (use_amcl=True) — odom alone has too much error
5097b92 Temporarily disable AMCL for shortfall diagnostic (use_amcl=False -> raw /odom)
3ad44a1 Bump cell size 70 cm -> 90 cm; ASSIGN_DIST_M -> 1.05 (still undershooting)
8c73fa9 Bump cell size 60 cm -> 70 cm; ASSIGN_DIST_M -> 0.85
ab98769 Bump cell size 50 cm -> 60 cm; ASSIGN_DIST_M 0.65 -> 0.75 to compensate undershoot
f5b744f Cell size back to 50 cm; ASSIGN_DIST_M -> 0.65
f056d4f Disable AMCL (use_amcl=False); 50 cm cells via raw /odom only
fba26df Reset detection window AFTER rotation completes, not before
5fefd2e Fix non-square grid action bug: swap n/m in planning/labeling calls
e24ba86 Move red marker prior to cell 3, yellow to cell 4 (green stays at 9)
b285ad6 Reconfigure for 6x3 grid: 0.5m cells, beliefs at cells 2/5/9, p_h=4
f80af53 Tighten yellow HSV: S_lo 60->140, V_lo 110->150 (reject white objects)
```

## Open issues / known limitations as of this snapshot

- **Tie-breaking** in Value_iteration uses `random.choice` when multiple actions share argmax. The user has observed odd behavior when ties occur in symmetric belief states. A heading-aware tie-break (prefer current heading to save a rotation) was suggested but not implemented.
- **Wheel-encoder calibration**: the robot's `/odom` slightly under-reports motion. AMCL compensates while running, but if AMCL drops out the planner sees inflated distances.
- **HSV overlaps**: latest in-situ tuning has small hue overlaps (yellow H 11-27 vs red H 0-12; green H 26-42 vs yellow H 11-27). Saturation thresholds usually separate them, but watch the `[DETECT]` lines for cross-detection.
- **Soft-hint logic**: when far-away markers are seen during look-around, soft-update at `round(dist/CELL_SIZE)` cells ahead of the *currently-faced* neighbor, capped at `SOFT_MAX_CELLS = 3`. Doesn't touch `perceived_labels`, only nudges `belief`.

## How to brief a fresh assistant chat

Open this file. Tell the new assistant: "Read CHAT_CONTEXT.md for project background. We're working on the TurtleBot3 LTL-planning project. <state your specific issue or change>."
