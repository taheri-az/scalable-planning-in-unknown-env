"""
Live grid visualization for the look-around planner.

Runs a Tkinter window in a background thread so it never blocks the planner
loop. The planner pushes state updates via `update(...)`; the window redraws
itself on its own timer by draining a thread-safe queue.

Shows, for an n×m grid:
  - the robot's current cell (highlighted, with a heading arrow),
  - discovered/perceived labels per cell (sticky observations),
  - the current belief per cell (dominant label + probability, plus a faint
    tint by dominant atomic),
  - the DFA state and step counter in a status strip.

Usage from the planner:
    from live_viz import LiveViz
    viz = LiveViz(n, m, atomics)          # opens the window (background thread)
    viz.update(robot_cell=0, heading='right',
               perceived=perceived_labels, belief=belief,
               dfa_state='1', step=3)
    ...
    viz.close()

If Tkinter or a display is unavailable the LiveViz becomes a no-op so the
planner still runs headless (e.g. over SSH without -X).
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

    def close(self):
        pass


class LiveViz:
    def __new__(cls, *a, **k):
        if not _TK_OK:
            print("[viz] Tkinter unavailable; running without visualization.")
            return _NullViz()
        return super().__new__(cls)

    def __init__(self, n, m, atomics, cell_px=96, title="Robot — live grid"):
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
        try:
            while True:
                msg = self._q.get_nowait()
                if msg.get("__close__"):
                    self._on_close()
                    return
                self._state.update(msg)
                dirty = True
        except queue.Empty:
            pass
        if dirty:
            self._redraw()
        if not self._closing:
            self.root.after(80, self._tick)

    # ---------- drawing ----------
    def _cell_fill(self, cell):
        """(bg, dominant_label_or_None, prob) using perceived first, then belief."""
        perceived = self._state["perceived"] or {}
        belief = self._state["belief"]
        emp = _empty_label(self.atomics)

        lbl = perceived.get(cell)
        if lbl is not None and lbl != emp:
            return self._atom_color(lbl), lbl, 1.0
        if belief is not None:
            dist = belief[cell]
            best = max(dist, key=lambda pl: pl[0])  # (prob, label)
            p, blabel = best
            if blabel != emp and p > 1.0 / (2 ** len(self.atomics)) + 1e-9:
                return self._atom_color(blabel, dim=True), blabel, p
        if lbl == emp:
            return GRID_EMPTYLBL, emp, 1.0
        return GRID_EMPTY, None, 0.0

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
        perceived = self._state["perceived"] or {}
        emp = _empty_label(self.atomics)

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
                fill, label, prob = self._cell_fill(cell)
                short = _label_short(label, self.atomics) if label else None
                seen = (perceived.get(cell) is not None
                        and perceived.get(cell) == label and label != emp)

                fg = _ideal_text_color(fill)
                # Observed (sticky) non-empty cells get a brighter border + a
                # solid fill; belief-only cells are faint; empties are flat.
                if seen:
                    self._rounded_rect(c, x0, y0, x1, y1, 12, fill=fill,
                                       outline=ROBOT_RING, width=3)
                else:
                    self._rounded_rect(c, x0, y0, x1, y1, 12, fill=fill,
                                       outline=BORDER, width=1)

                # Cell index, small, top-left.
                c.create_text(x0 + 9, y0 + 8, text=str(cell), anchor="nw",
                              fill=fg, font=self.font_idx)

                # ── Label rendering ──
                if short and short != "empty":
                    # Big atomic name in the center.
                    c.create_text(cx, cy - 4, text=short.upper(), fill=fg,
                                  font=self.font_cell)
                    if seen:
                        # A confirmed observation: checkmark badge.
                        c.create_text(cx, cy + 18, text="✓ observed", fill=fg,
                                      font=self.font_small)
                    else:
                        # Belief only: show probability + a thin mass bar.
                        c.create_text(cx, cy + 18,
                                      text=f"{int(round(prob * 100))}%",
                                      fill=fg, font=self.font_small)
                        bw = (x1 - x0) - 24
                        by = y1 - 12
                        c.create_rectangle(x0 + 12, by, x0 + 12 + bw, by + 4,
                                           fill=BORDER, outline="")
                        c.create_rectangle(x0 + 12, by,
                                           x0 + 12 + int(bw * min(1.0, prob)),
                                           by + 4, fill=fg, outline="")
                elif perceived.get(cell) == emp:
                    # Visited & confirmed empty.
                    c.create_text(cx, cy, text="·", fill=MUTED,
                                  font=self.font_cell)

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
    # Tiny self-test: a fake robot wandering a 6×3 grid.
    import time
    import numpy as np
    from labeling import assign_probabilities_g3

    atomics = ["a", "b", "c"]
    n, m = 6, 3
    belief = assign_probabilities_g3(n, m, atomics, initial_belief={
        3: {"a && !b && !c": 0.8, "!a && !b && !c": 0.2},
        4: {"!a && b && !c": 0.8, "!a && !b && !c": 0.2},
        9: {"!a && !b && c": 0.8, "!a && !b && !c": 0.2},
    })
    viz = LiveViz(n, m, atomics)

    def fake_robot():
        perceived = {0: "!a && !b && !c"}
        path = [0, 3, 4, 7, 10, 9]
        headings = ["right", "down", "right", "down", "down", "left"]
        for i, (cell, hd) in enumerate(zip(path, headings)):
            if cell == 3:
                perceived[3] = "a && !b && !c"
            if cell == 4:
                perceived[4] = "!a && b && !c"
            if cell == 9:
                perceived[9] = "!a && !b && c"
            viz.update(robot_cell=cell, heading=hd, perceived=perceived,
                       belief=belief, dfa_state=str(i), step=i + 1)
            time.sleep(1.2)
        time.sleep(2)

    # Tk on the main thread; the fake robot drives from a worker thread.
    viz.run(fake_robot)
