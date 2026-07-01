"""
Live grid visualization for the look-around planner.

Tkinter/Tcl is not thread-safe, so the Tk event loop must own the MAIN thread.
`LiveViz.run(worker_fn)` runs the Tk loop on the calling (main) thread and
launches the planner `worker_fn` in a background thread. The worker pushes
state updates via `update(...)`; the window redraws on its own timer by
draining a thread-safe queue. (Creating Tk in a side thread instead caused
"Tcl_AsyncDelete: async handler deleted by the wrong thread" at shutdown.)

Shows, for an n×m grid:
  - the robot's current cell as a centered glowing disc with a heading arrow,
  - observed/perceived labels per cell (sticky; bright ring + "✓ observed"),
  - the current belief per un-observed cell: the top-2 labels with their
    probabilities and mass bars, tinted by the dominant atomic,
  - the DFA state and step counter in a status strip.

Usage from the planner:
    from live_viz import LiveViz
    viz = LiveViz(n, m, atomics)          # no window yet
    def planner():
        ...
        viz.update(robot_cell=0, heading='right',
                   perceived=perceived_labels, belief=belief,
                   dfa_state='1', step=3)
        ...
    viz.run(planner)                       # Tk on main thread; planner in worker

If Tkinter or a display is unavailable, run() just executes the worker with no
GUI so the planner still runs headless (e.g. over SSH without -X).
"""

import queue
import threading

try:
    import tkinter as tk
    from tkinter import font as tkfont
    _TK_OK = True
except Exception:
    _TK_OK = False


# Palette (kept in sync visually with build_belief_gui.py).
BG          = "#0e1320"
PANEL       = "#161e30"
CARD        = "#212c44"
FG          = "#f2f5fb"
MUTED       = "#7e8aa6"
ROBOT       = "#ffd24a"
ROBOT_RING  = "#fff1b8"
GRID_EMPTY  = "#1c2438"
GRID_EMPTYLBL = "#46506a"
BORDER      = "#2a3653"
ATOM_COLORS = ["#ff5d6c", "#f5b73c", "#3ed492", "#6c8cff", "#c07bff", "#3fd0d6"]

HEADING_ARROW = {"up": "▲", "down": "▼", "left": "◀", "right": "▶", None: "●"}


def _ideal_text_color(hex_bg):
    hex_bg = hex_bg.lstrip("#")[:6]
    if len(hex_bg) != 6:
        return FG
    r, g, b = (int(hex_bg[i:i + 2], 16) for i in (0, 2, 4))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "#13171f" if luma > 150 else "#ffffff"


def _label_short(label, atomics):
    terms = [t.strip() for t in label.split('&&')]
    trues = [t for t in terms if not t.startswith('!')]
    return " ".join(trues) if trues else "empty"


def _empty_label(atomics):
    return ' && '.join(f'!{a}' for a in atomics)


class _NullViz:
    """No-op stand-in when Tk/display is unavailable."""
    def update(self, **kw):
        pass

    def flush(self, timeout=1.0):
        pass

    def run(self, worker_fn):
        worker_fn()

    def close(self):
        pass


class LiveViz:
    def __new__(cls, *a, **k):
        if not _TK_OK:
            print("[viz] Tkinter unavailable; running without visualization.")
            return _NullViz()
        return super().__new__(cls)

    def __init__(self, n, m, atomics, cell_px=96, title="Robot — live grid",
                 belief_pkl="belief.pkl"):
        self.n, self.m, self.atomics = n, m, atomics
        self.cell_px = cell_px
        self.title = title
        self._q = queue.Queue()
        self._state = {
            "robot_cell": 0, "heading": None, "perceived": {},
            "belief": None, "dfa_state": "?", "step": 0,
        }
        self._closing = False
        self.root = None
        # Seed the initial belief from the same pickle the planner uses, so the
        # very first frame already shows the prior belief. Live viz.update(
        # belief=...) calls then overwrite this as the robot moves.
        if belief_pkl:
            self._load_initial_belief(belief_pkl)

    def _load_initial_belief(self, path):
        try:
            import os
            import pickle
            if not os.path.exists(path):
                return
            with open(path, "rb") as f:
                data = pickle.load(f)
            if (data.get("n"), data.get("m")) != (self.n, self.m):
                print(f"[viz] {path} is for grid {data.get('n')}x{data.get('m')}, "
                      f"viz is {self.n}x{self.m}; ignoring for initial belief.")
                return
            if set(data.get("atomics", [])) != set(self.atomics):
                print(f"[viz] {path} atomics {data.get('atomics')} != "
                      f"{self.atomics}; ignoring for initial belief.")
                return
            self._state["belief"] = data["belief"]
            print(f"[viz] seeded initial belief from {path}.")
        except Exception as e:
            print(f"[viz] could not load initial belief from {path}: {e}")

    # ---------- public API (called from the planner/worker thread) ----------
    def update(self, **kw):
        """Push a partial state update. Thread-safe. Recognized keys:
        robot_cell, heading, perceived, belief, dfa_state, step."""
        if self._closing:
            return
        try:
            self._q.put_nowait(dict(kw))
        except Exception:
            pass

    def flush(self, timeout=1.0):
        """Block until all queued updates have been drained AND redrawn, so the
        caller can guarantee the map reflects the latest state before doing
        something else (e.g. commanding a move). Cheap no-op if headless."""
        if self._closing or self.root is None:
            return
        ev = threading.Event()
        try:
            self._q.put_nowait({"__flush__": ev})
        except Exception:
            return
        ev.wait(timeout=timeout)

    def close(self):
        self._closing = True
        try:
            self._q.put_nowait({"__close__": True})
        except Exception:
            pass

    def run(self, worker_fn):
        """Run the Tk event loop on the CALLING thread (must be the main
        thread) and run `worker_fn` (the planner) in a background thread.

        Tkinter/Tcl is not thread-safe: the interpreter that creates the Tk
        root must also be the one that runs the mainloop and tears it down.
        Creating Tk in a side thread (and letting the main thread exit) is
        exactly what triggers 'Tcl_AsyncDelete: async handler deleted by the
        wrong thread' on shutdown. So we keep Tk on the main thread and push
        the planner to a worker thread instead.

        When `worker_fn` returns (or raises), the window is closed and `run`
        returns. Exceptions from the worker are re-raised here.
        """
        self._worker_exc = None

        def _worker():
            try:
                worker_fn()
            except BaseException as e:           # noqa: BLE001 - surfaced below
                self._worker_exc = e
            finally:
                self.close()

        if not self._open_window():
            # No display: just run the worker synchronously, no GUI.
            _worker()
            if self._worker_exc is not None:
                raise self._worker_exc
            return

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            self.root.mainloop()
        finally:
            t.join(timeout=2.0)
        if self._worker_exc is not None:
            raise self._worker_exc

    # ---------- window (runs on main thread via run()) ----------
    def _open_window(self):
        try:
            self.root = tk.Tk()
        except Exception as e:
            print(f"[viz] could not open window ({e}); continuing headless.")
            self.root = None
            return False
        self.root.title(self.title)
        self.root.configure(bg=BG)

        self.font_title = tkfont.Font(family="DejaVu Sans", size=12, weight="bold")
        self.font_cell  = tkfont.Font(family="DejaVu Sans", size=16, weight="bold")
        self.font_lbl   = tkfont.Font(family="DejaVu Sans", size=11, weight="bold")
        self.font_small = tkfont.Font(family="DejaVu Sans", size=9)
        self.font_idx   = tkfont.Font(family="DejaVu Sans", size=8)

        # Give the canvas a bit of slack around the grid so it can be centered
        # (and stays centered if the window is resized).
        self._margin = 24
        cw = self.m * self.cell_px + 2 * self._margin
        ch = self.n * self.cell_px + 2 * self._margin
        self.canvas = tk.Canvas(self.root, width=cw, height=ch, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(14, 4))
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.status = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                               font=self.font_title, anchor="w")
        self.status.pack(fill="x", padx=16, pady=(0, 12))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(60, self._tick)
        return True

    def _on_close(self):
        self._closing = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _tick(self):
        # Drain all pending updates, keeping the latest of each key.
        dirty = False
        flush_events = []
        try:
            while True:
                msg = self._q.get_nowait()
                if msg.get("__close__"):
                    self._on_close()
                    return
                if "__flush__" in msg:
                    # Redraw now so the pending state is painted, then release
                    # the waiter. Handle after draining the rest of the queue.
                    flush_events.append(msg["__flush__"])
                    dirty = True
                    continue
                self._state.update(msg)
                dirty = True
        except queue.Empty:
            pass
        if dirty:
            self._redraw()
            self.canvas.update_idletasks()   # force the paint to hit the screen
        for ev in flush_events:
            ev.set()
        if not self._closing:
            self.root.after(80, self._tick)

    # ---------- drawing ----------
    def _cell_info(self, cell):
        """Return a dict describing what to draw for `cell`:
            kind: 'observed' | 'empty' | 'belief' | 'blank'
            fill: background hex
            label: dominant non-empty label (or None)
            top:   list of (prob, short_label) for the top belief entries
                   (always includes the empty label so the belief is visible).
        Perceived (sticky) observations take priority over belief.
        """
        perceived = self._state["perceived"] or {}
        belief = self._state["belief"]
        emp = _empty_label(self.atomics)

        lbl = perceived.get(cell)
        if lbl is not None and lbl != emp:
            return {"kind": "observed", "fill": self._atom_color(lbl),
                    "label": lbl, "top": [(1.0, _label_short(lbl, self.atomics))]}
        if lbl == emp:
            return {"kind": "empty", "fill": GRID_EMPTYLBL, "label": None,
                    "top": [(1.0, "empty")]}

        # Un-observed cell: show the actual belief.
        if belief is not None:
            dist = sorted(belief[cell], key=lambda pl: -pl[0])
            top = [(float(p), _label_short(l, self.atomics)) for p, l in dist[:2]]
            best_l = dist[0][1]
            if best_l != emp:
                # Dominant belief is a real marker → tint by its atomic.
                return {"kind": "belief", "fill": self._atom_color(best_l, dim=True),
                        "label": best_l, "top": top}
            # Dominant belief is empty, but a secondary marker may still matter:
            # tint faintly by the strongest non-empty belief if it's non-trivial.
            nonempty = [(p, l) for p, l in dist if l != emp]
            if nonempty and nonempty[0][0] > 0.08:
                fill = self._dim2(self._atom_color(nonempty[0][1]))
            else:
                fill = GRID_EMPTY
            return {"kind": "belief", "fill": fill, "label": None, "top": top}
        return {"kind": "blank", "fill": GRID_EMPTY, "label": None, "top": []}

    def _short_to_label(self, short):
        """Inverse of _label_short: 'a b' -> 'a && b', 'empty' -> empty label."""
        if short in ("empty", "", "·"):
            return _empty_label(self.atomics)
        return " && ".join(short.split())

    def _atom_color(self, label, dim=False):
        terms = [t.strip() for t in label.split('&&')]
        trues = [t for t in terms if not t.startswith('!')]
        if trues and trues[0] in self.atomics:
            c = ATOM_COLORS[self.atomics.index(trues[0]) % len(ATOM_COLORS)]
            return self._dim(c) if dim else c
        return GRID_EMPTYLBL

    @staticmethod
    def _dim(hexc):
        hexc = hexc.lstrip("#")
        r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
        # Blend ~45% toward the empty-cell colour for a faint belief tint.
        er, eg, eb = (int(GRID_EMPTY.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        f = 0.55
        return "#%02x%02x%02x" % (int(r * f + er * (1 - f)),
                                  int(g * f + eg * (1 - f)),
                                  int(b * f + eb * (1 - f)))

    @staticmethod
    def _dim2(hexc):
        """Stronger blend toward the empty-cell colour — a very faint tint used
        for cells where the dominant belief is empty but a secondary marker
        still has some mass."""
        hexc = hexc.lstrip("#")
        r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
        er, eg, eb = (int(GRID_EMPTY.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        f = 0.28
        return "#%02x%02x%02x" % (int(r * f + er * (1 - f)),
                                  int(g * f + eg * (1 - f)),
                                  int(b * f + eb * (1 - f)))

    @staticmethod
    def _rounded_rect(c, x0, y0, x1, y1, rad, **kw):
        """Draw a rounded rectangle as a smoothed polygon."""
        rad = min(rad, (x1 - x0) / 2, (y1 - y0) / 2)
        pts = [
            x0 + rad, y0, x1 - rad, y0, x1, y0, x1, y0 + rad,
            x1, y1 - rad, x1, y1, x1 - rad, y1, x0 + rad, y1,
            x0, y1, x0, y1 - rad, x0, y0 + rad, x0, y0,
        ]
        return c.create_polygon(pts, smooth=True, **kw)

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        s = self.cell_px
        gap = 5
        robot = self._state["robot_cell"]

        # ── Center the grid in the canvas ──
        cw = c.winfo_width()  or (self.m * s + 2 * self._margin)
        ch = c.winfo_height() or (self.n * s + 2 * self._margin)
        grid_w, grid_h = self.m * s, self.n * s
        ox = max(self._margin, (cw - grid_w) // 2)
        oy = max(self._margin, (ch - grid_h) // 2)

        for r in range(self.n):
            for col in range(self.m):
                cell = r * self.m + col
                x0 = ox + col * s
                y0 = oy + r * s
                x1, y1 = x0 + s - gap, y0 + s - gap
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                info = self._cell_info(cell)
                fill = info["fill"]
                fg = _ideal_text_color(fill)

                # Observed (sticky) non-empty cells get a bright ring; every
                # other cell a thin border.
                if info["kind"] == "observed":
                    self._rounded_rect(c, x0, y0, x1, y1, 12, fill=fill,
                                       outline=ROBOT_RING, width=3)
                else:
                    self._rounded_rect(c, x0, y0, x1, y1, 12, fill=fill,
                                       outline=BORDER, width=1)

                # Cell index, small, top-left.
                c.create_text(x0 + 9, y0 + 8, text=str(cell), anchor="nw",
                              fill=fg, font=self.font_idx)

                # ── Content per cell kind ──
                if info["kind"] == "observed":
                    short = _label_short(info["label"], self.atomics)
                    c.create_text(cx, cy - 6, text=short.upper(), fill=fg,
                                  font=self.font_cell)
                    c.create_text(cx, cy + 16, text="✓ observed", fill=fg,
                                  font=self.font_small)

                elif info["kind"] == "empty":
                    c.create_text(cx, cy - 4, text="empty", fill=fg,
                                  font=self.font_small)
                    c.create_text(cx, cy + 12, text="✓ seen", fill=fg,
                                  font=self.font_small)

                elif info["kind"] == "belief" and info["top"]:
                    # Show the belief itself: top-1 label big, then up to two
                    # (label prob%) rows with mass bars so the belief is
                    # readable at a glance and updates each step.
                    top = info["top"]
                    head_short = top[0][1]
                    c.create_text(cx, y0 + 26,
                                  text=(head_short.upper()
                                        if head_short != "empty" else "·"),
                                  fill=fg, font=self.font_cell)
                    row_y = cy + 6
                    bw = (x1 - x0) - 24
                    for p, sh in top:
                        c.create_text(x0 + 12, row_y, anchor="w",
                                      text=f"{sh} {int(round(p * 100))}%",
                                      fill=fg, font=self.font_small)
                        by = row_y + 10
                        c.create_rectangle(x0 + 12, by, x0 + 12 + bw, by + 3,
                                           fill=BORDER, outline="")
                        c.create_rectangle(
                            x0 + 12, by, x0 + 12 + int(bw * min(1.0, p)),
                            by + 3, fill=(self._atom_color(
                                self._short_to_label(sh)) if sh != "empty"
                                else MUTED), outline="")
                        row_y += 22

                # ── Robot marker: centered glowing disc with heading arrow ──
                if cell == robot:
                    rad = s * 0.30
                    c.create_oval(cx - rad - 4, cy - rad - 4,
                                  cx + rad + 4, cy + rad + 4,
                                  outline=ROBOT_RING, width=2)
                    c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                  fill=ROBOT, outline=ROBOT_RING, width=2)
                    arrow = HEADING_ARROW.get(self._state["heading"], "●")
                    c.create_text(cx, cy, text=arrow, fill="#1a1a1a",
                                  font=self.font_cell)

        st = self._state
        self.status.config(
            text=f"step {st['step']}    ·    robot @ cell {robot}    ·    "
                 f"dfa = {st['dfa_state']}    ·    "
                 f"heading {st['heading'] or '—'}")


if __name__ == "__main__":
    # Self-test: a fake robot wandering a grid. If belief.pkl exists (built by
    # build_belief_gui.py), use ITS grid size / atomics / belief so you can
    # double-check the belief.pkl seeding + live-update path end to end.
    # Otherwise fall back to a built-in 6×3 example.
    import os
    import time
    import pickle
    import argparse
    from labeling import assign_probabilities_g3

    ap = argparse.ArgumentParser()
    ap.add_argument("--belief", default="belief.pkl",
                    help="belief pickle to load (default: belief.pkl)")
    args = ap.parse_args()

    if os.path.exists(args.belief):
        with open(args.belief, "rb") as f:
            data = pickle.load(f)
        n, m, atomics = data["n"], data["m"], data["atomics"]
        belief = data["belief"]
        print(f"[demo] loaded {args.belief}: grid {n}x{m}, atomics {atomics}")
    else:
        print(f"[demo] {args.belief} not found; using built-in 6×3 example.")
        atomics = ["a", "b", "c"]
        n, m = 6, 3
        belief = assign_probabilities_g3(n, m, atomics, initial_belief={
            3: {"a && !b && !c": 0.8, "!a && !b && !c": 0.2},
            4: {"!a && b && !c": 0.8, "!a && !b && !c": 0.2},
            9: {"!a && !b && c": 0.8, "!a && !b && !c": 0.2},
        })

    # LiveViz also seeds the initial belief from the pickle on its own; passing
    # belief_pkl here keeps the very first frame consistent with `belief`.
    viz = LiveViz(n, m, atomics, belief_pkl=args.belief)
    emp = _empty_label(atomics)

    def _cell_label(cell):
        """Which sticky label a cell reveals when visited: the dominant
        non-empty belief label if any, else empty. This lets the demo 'observe'
        exactly what your belief.pkl expects at each cell."""
        dist = sorted(belief[cell], key=lambda pl: -pl[0])
        p, lbl = dist[0]
        return lbl if (lbl != emp and p > 0.4) else emp

    def fake_robot():
        # Walk a snake path over the whole grid so every cell gets visited and
        # its belief collapses to the observed label as we pass.
        perceived = {0: _cell_label(0)}
        path, headings = [], []
        cur = 0
        for r in range(n):
            cols = range(m) if r % 2 == 0 else range(m - 1, -1, -1)
            for col in cols:
                nxt = r * m + col
                if nxt == cur and not path:
                    path.append(nxt); headings.append(None); continue
                # heading from cur -> nxt
                dr, dc = divmod(nxt, m)[0] - divmod(cur, m)[0], (nxt % m) - (cur % m)
                hd = ("right" if dc > 0 else "left" if dc < 0 else
                      "down" if dr > 0 else "up" if dr < 0 else None)
                path.append(nxt); headings.append(hd); cur = nxt

        for i, (cell, hd) in enumerate(zip(path, headings)):
            perceived[cell] = _cell_label(cell)   # observe on entry
            viz.update(robot_cell=cell, heading=hd, perceived=perceived,
                       belief=belief, dfa_state=str(i), step=i + 1)
            time.sleep(1.0)
        time.sleep(2)

    # Tk on the main thread; the fake robot drives from a worker thread.
    viz.run(fake_robot)
