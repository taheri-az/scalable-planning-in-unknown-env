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
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Don't block long; if the window fails to come up we just no-op draws.
        self._ready.wait(timeout=3.0)

    # ---------- public API (called from the planner thread) ----------
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

    # ---------- window thread ----------
    def _run(self):
        try:
            self.root = tk.Tk()
        except Exception as e:
            print(f"[viz] could not open window ({e}); continuing headless.")
            # Drain queue forever so the planner's puts never block.
            self._ready.set()
            try:
                while True:
                    self._q.get()
            except Exception:
                return
        self.root.title(self.title)
        self.root.configure(bg=BG)

        self.font_title = tkfont.Font(family="DejaVu Sans", size=12, weight="bold")
        self.font_cell  = tkfont.Font(family="DejaVu Sans", size=13, weight="bold")
        self.font_lbl   = tkfont.Font(family="DejaVu Sans", size=9)
        self.font_small = tkfont.Font(family="DejaVu Sans", size=8)

        w = self.m * self.cell_px + 32
        h = self.n * self.cell_px + 86
        self.canvas = tk.Canvas(self.root, width=w, height=h - 40, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(padx=16, pady=(14, 4))
        self.status = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                               font=self.font_title, anchor="w")
        self.status.pack(fill="x", padx=16, pady=(0, 12))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._ready.set()
        self.root.after(60, self._tick)
        self.root.mainloop()

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

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        s = self.cell_px
        gap = 4
        robot = self._state["robot_cell"]
        perceived = self._state["perceived"] or {}
        emp = _empty_label(self.atomics)

        for r in range(self.n):
            for col in range(self.m):
                cell = r * self.m + col
                x0 = 16 + col * s
                y0 = 4 + r * s
                x1, y1 = x0 + s - gap, y0 + s - gap
                fill, label, prob = self._cell_fill(cell)
                outline = BORDER
                c.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline,
                                   width=1)
                fg = _ideal_text_color(fill)

                # Cell index (top-left).
                c.create_text(x0 + 8, y0 + 6, text=str(cell), anchor="nw",
                              fill=fg, font=self.font_small)

                # Label + probability (center).
                if label is not None:
                    short = _label_short(label, self.atomics)
                    is_sticky = perceived.get(cell) not in (None,)
                    txt = short if short != "empty" else "·"
                    c.create_text((x0 + x1) / 2, (y0 + y1) / 2 - 6, text=txt,
                                  fill=fg, font=self.font_cell)
                    if short != "empty":
                        sub = ("seen" if (is_sticky and perceived.get(cell) == label
                                          and label != emp)
                               else f"{int(round(prob * 100))}%")
                        c.create_text((x0 + x1) / 2, (y0 + y1) / 2 + 14,
                                      text=sub, fill=fg, font=self.font_small)

                # Robot marker.
                if cell == robot:
                    c.create_rectangle(x0, y0, x1, y1, outline=ROBOT_RING,
                                       width=3)
                    arrow = HEADING_ARROW.get(self._state["heading"], "●")
                    c.create_text(x1 - 12, y1 - 12, text=arrow, fill=ROBOT,
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
    viz.close()
