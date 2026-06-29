"""
GUI belief builder for the look-around planner (Tkinter).

A windowed version of build_belief.py. Flow:
  1. Set grid rows (n) / cols (m) and the atomic propositions, click "Build grid".
  2. Click a cell in the grid. Pick a label (combination of TRUE atomics) from
     the list, set a probability with the slider, click "Add to cell". Repeat —
     the remaining mass auto-shows on the empty label.
  3. Click "Save belief.pkl". The zeta floor + renormalization is applied via
     assign_probabilities_g3 (the same routine the planner uses), and the result
     is pickled to belief.pkl.

Cell numbering matches the planner: cell = row * m + col  (stride = m).

Run on the Pi over SSH with X-forwarding:
    ssh -X ubuntu@192.168.0.145
    cd ~/scalable-planning-in-unknown-env
    python3 build_belief_gui.py

If you get "no display name and no $DISPLAY", you forgot the -X on ssh.
"""

import sys
import pickle
import itertools

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from tkinter import font as tkfont
except Exception as e:  # pragma: no cover - environment without Tk
    print("Tkinter not available:", e)
    print("On the Pi: sudo apt-get install python3-tk, and ssh with -X.")
    sys.exit(1)

from labeling import assign_probabilities_g3

ZETA_DEFAULT = 0.02

# ───────────────────────── theme palette ─────────────────────────
BG          = "#0e1320"   # window background (deep indigo)
HEADER_BG   = "#151c2e"   # header band
PANEL       = "#161e30"   # panel background
CARD        = "#212c44"   # card / listbox background
CARD_HOVER  = "#2b3858"
FG          = "#f2f5fb"   # primary text
MUTED       = "#7e8aa6"   # secondary text
ACCENT      = "#6c8cff"   # primary accent (periwinkle)
ACCENT_DK   = "#5573ef"
ACCENT_HOV  = "#88a3ff"
OK_GREEN    = "#3ddc97"
WARN_RED    = "#ff6b6b"
GRID_EMPTY  = "#1c2438"   # unassigned cell
GRID_EMPTYLBL = "#46506a"  # cell whose dominant label is "empty"
SEL_RING    = "#9fb6ff"   # selected cell ring (accent glow)
TRACK       = "#212c44"
BORDER      = "#2a3653"

# Per-atomic accent colours (a=red, b=yellow, c=green by convention).
ATOM_COLORS = ["#ff5d6c", "#f5b73c", "#3ed492", "#6c8cff", "#c07bff", "#3fd0d6"]


def all_labels(atomics):
    return [
        ' && '.join(atomics[i] if bits[i] else f'!{atomics[i]}'
                    for i in range(len(atomics)))
        for bits in itertools.product([0, 1], repeat=len(atomics))
    ]


def empty_label(atomics):
    return ' && '.join(f'!{a}' for a in atomics)


def label_short(label, atomics):
    """Human-friendly short name: the TRUE atomics, or 'empty'."""
    terms = [t.strip() for t in label.split('&&')]
    trues = [t for t in terms if not t.startswith('!')]
    return " ".join(trues) if trues else "empty"


def _ideal_text_color(hex_bg):
    """Pick black or white text for legibility on a given hex background."""
    hex_bg = hex_bg.lstrip("#")[:6]
    if len(hex_bg) != 6:
        return FG
    r, g, b = (int(hex_bg[i:i + 2], 16) for i in (0, 2, 4))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "#13171f" if luma > 150 else "#ffffff"


class HoverButton(tk.Button):
    """Flat button with a hover color swap — gives a non-'pixel' feel."""
    def __init__(self, parent, hover_bg, base_bg, **kw):
        super().__init__(parent, **kw)
        self._hover_bg, self._base_bg = hover_bg, base_bg
        self.config(bg=base_bg, activebackground=hover_bg, relief="flat",
                    bd=0, highlightthickness=0, cursor="hand2")
        self.bind("<Enter>", lambda e: self.config(bg=self._hover_bg))
        self.bind("<Leave>", lambda e: self.config(bg=self._base_bg))


class BeliefGUI:
    def __init__(self, root):
        self.root = root
        root.title("Belief Builder")
        root.configure(bg=BG)
        root.minsize(1040, 760)
        self._center(1120, 820)

        self.n = 6
        self.m = 3
        self.atomics = ["a", "b", "c"]
        self.zeta = ZETA_DEFAULT
        self.initial_belief = {}          # {cell: {label_string: prob}}
        self.selected_cell = None
        self.cell_buttons = {}

        self._init_fonts()
        self._init_style()
        self._build_header()
        self._build_body()
        self.rebuild_grid()

    def _center(self, w, h):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = (sw - w) // 2, max(0, (sh - h) // 3)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _init_fonts(self):
        fam = "DejaVu Sans"
        self.font_h1   = tkfont.Font(family=fam, size=17, weight="bold")
        self.font_h2   = tkfont.Font(family=fam, size=11, weight="bold")
        self.font_lbl  = tkfont.Font(family=fam, size=9,  weight="bold")
        self.font_body = tkfont.Font(family=fam, size=10)
        self.font_mono = tkfont.Font(family="DejaVu Sans Mono", size=10)
        self.font_cell = tkfont.Font(family=fam, size=13, weight="bold")
        self.font_celllbl = tkfont.Font(family=fam, size=9)

    def _init_style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=PANEL, foreground=FG, font=self.font_body,
                     borderwidth=0)
        st.configure("TFrame", background=PANEL)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=PANEL, foreground=FG)
        st.configure("Muted.TLabel", background=PANEL, foreground=MUTED,
                     font=self.font_lbl)
        st.configure("H2.TLabel", background=PANEL, foreground=FG, font=self.font_h2)
        st.configure("TEntry", fieldbackground=CARD, foreground=FG,
                     bordercolor=CARD, lightcolor=CARD, darkcolor=CARD,
                     insertcolor=FG, padding=6, relief="flat")
        st.configure("Horizontal.TScale", background=PANEL, troughcolor=TRACK,
                     bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT_DK)
        st.map("Horizontal.TScale", background=[("active", PANEL)])

    # ---------------- header / config bar ----------------
    def _build_header(self):
        # Banded header with a thin accent stripe down the left.
        band = tk.Frame(self.root, bg=HEADER_BG)
        band.grid(row=0, column=0, sticky="ew")
        self.root.columnconfigure(0, weight=1)
        stripe = tk.Frame(band, bg=ACCENT, width=5)
        stripe.pack(side="left", fill="y")
        htxt = tk.Frame(band, bg=HEADER_BG)
        htxt.pack(side="left", fill="x", expand=True, padx=22, pady=18)
        tk.Label(htxt, text="◆  Belief Builder", bg=HEADER_BG, fg=FG,
                 font=self.font_h1).pack(anchor="w")
        tk.Label(htxt,
                 text="Click a cell · pick a label · set its probability.  "
                      "The ζ-floor is applied automatically on save.",
                 bg=HEADER_BG, fg=MUTED, font=self.font_body).pack(
            anchor="w", pady=(3, 0))

        cfg = tk.Frame(self.root, bg=BORDER)   # 1px border via padding trick
        cfg.grid(row=1, column=0, sticky="ew", padx=22, pady=(16, 0))
        body = tk.Frame(cfg, bg=PANEL)
        body.pack(fill="x", padx=1, pady=1)
        inner = tk.Frame(body, bg=PANEL)
        inner.pack(fill="x", padx=14, pady=12)

        self.n_var = tk.StringVar(value=str(self.n))
        self.m_var = tk.StringVar(value=str(self.m))
        self.atoms_var = tk.StringVar(value=" ".join(self.atomics))
        self.zeta_var = tk.StringVar(value=str(self.zeta))

        def field(label, var, width, col):
            tk.Label(inner, text=label, bg=PANEL, fg=MUTED,
                     font=self.font_lbl).grid(row=0, column=col,
                                              padx=(0 if col == 0 else 18, 7),
                                              sticky="w")
            ttk.Entry(inner, textvariable=var, width=width).grid(
                row=0, column=col + 1)

        field("ROWS (n)", self.n_var, 4, 0)
        field("COLS (m)", self.m_var, 4, 2)
        field("ATOMICS", self.atoms_var, 16, 4)
        field("ζ", self.zeta_var, 6, 6)

        HoverButton(inner, ACCENT_HOV, ACCENT, text="Build grid",
                    fg="#ffffff", font=self.font_h2, padx=16, pady=7,
                    command=self.rebuild_grid).grid(row=0, column=8, padx=(20, 0))

    # ---------------- body: grid + side panel ----------------
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.grid(row=2, column=0, sticky="nsew", padx=22, pady=16)
        self.root.rowconfigure(2, weight=1)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # ── Left: grid card (1px border) ──
        left_b = tk.Frame(body, bg=BORDER)
        left_b.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left = tk.Frame(left_b, bg=PANEL)
        left.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(left, text="GRID", bg=PANEL, fg=MUTED,
                 font=self.font_lbl).pack(anchor="w", padx=18, pady=(16, 0))
        self.grid_frame = tk.Frame(left, bg=PANEL)
        self.grid_frame.pack(fill="both", expand=True, padx=16, pady=12)
        self.legend = tk.Frame(left, bg=PANEL)
        self.legend.pack(fill="x", padx=18, pady=(0, 16))

        # ── Right: editor panel (fixed width, actions near top) ──
        side_b = tk.Frame(body, bg=BORDER, width=340)
        side_b.grid(row=0, column=1, sticky="ns")
        side_b.pack_propagate(False)
        side = tk.Frame(side_b, bg=PANEL)
        side.pack(fill="both", expand=True, padx=1, pady=1)
        pad = tk.Frame(side, bg=PANEL)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        self.sel_label = tk.Label(pad, text="No cell selected", bg=PANEL, fg=FG,
                                  font=self.font_h2, anchor="w")
        self.sel_label.pack(fill="x", pady=(0, 12))

        tk.Label(pad, text="LABEL", bg=PANEL, fg=MUTED,
                 font=self.font_lbl).pack(anchor="w")
        self.label_list = tk.Listbox(
            pad, height=7, exportselection=False, bg=CARD, fg=FG,
            font=self.font_body, relief="flat", highlightthickness=0,
            selectbackground=ACCENT, selectforeground="#ffffff",
            activestyle="none", bd=0)
        self.label_list.pack(fill="x", pady=(4, 12))

        tk.Label(pad, text="PROBABILITY", bg=PANEL, fg=MUTED,
                 font=self.font_lbl).pack(anchor="w")
        prob_row = tk.Frame(pad, bg=PANEL)
        prob_row.pack(fill="x", pady=(4, 10))
        self.prob_value = tk.DoubleVar(value=0.80)
        self.prob_scale = ttk.Scale(prob_row, from_=0.0, to=1.0,
                                    orient="horizontal", variable=self.prob_value,
                                    command=self._on_prob_slide)
        self.prob_scale.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.prob_readout = tk.Label(prob_row, text="80%", bg=PANEL, fg=ACCENT,
                                     font=self.font_h2, width=5, anchor="e")
        self.prob_readout.pack(side="right")

        HoverButton(pad, ACCENT_HOV, ACCENT, text="＋  Add to cell",
                    fg="#ffffff", font=self.font_h2, pady=9,
                    command=self.add_to_cell).pack(fill="x", pady=(2, 16))

        tk.Label(pad, text="ASSIGNED AT THIS CELL", bg=PANEL, fg=MUTED,
                 font=self.font_lbl).pack(anchor="w")
        self.cell_dist = tk.Listbox(
            pad, height=6, exportselection=False, bg=CARD, fg=FG,
            font=self.font_mono, relief="flat", highlightthickness=0,
            selectbackground=ACCENT, selectforeground="#ffffff",
            activestyle="none", bd=0)
        self.cell_dist.pack(fill="x", pady=(4, 6))

        # Mass progress bar.
        self.bar_canvas = tk.Canvas(pad, height=6, bg=TRACK, highlightthickness=0,
                                    bd=0)
        self.bar_canvas.pack(fill="x", pady=(0, 4))
        self.remain_label = tk.Label(pad, text="", bg=PANEL, fg=MUTED,
                                     font=self.font_body, anchor="w")
        self.remain_label.pack(fill="x", pady=(0, 8))

        btns = tk.Frame(pad, bg=PANEL)
        btns.pack(fill="x")
        HoverButton(btns, CARD_HOVER, CARD, text="Remove", fg=FG,
                    font=self.font_body, pady=7,
                    command=self.remove_selected).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        HoverButton(btns, CARD_HOVER, CARD, text="Clear cell", fg=FG,
                    font=self.font_body, pady=7,
                    command=self.clear_cell).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        # Save area pinned to the bottom of the panel.
        save_area = tk.Frame(pad, bg=PANEL)
        save_area.pack(side="bottom", fill="x", pady=(16, 0))
        HoverButton(save_area, ACCENT_HOV, ACCENT, text="💾  Save belief.pkl",
                    fg="#ffffff", font=self.font_h2, pady=10,
                    command=self.save).pack(fill="x")
        HoverButton(save_area, CARD_HOVER, CARD, text="Save as…", fg=FG,
                    font=self.font_body, pady=7,
                    command=self.save_as).pack(fill="x", pady=(6, 0))

        # ── Status bar ──
        self.status = tk.Label(self.root, text="", bg=BG, fg=OK_GREEN,
                               font=self.font_body, anchor="w")
        self.status.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 12))

    def _on_prob_slide(self, _v):
        self.prob_readout.config(text=f"{int(round(self.prob_value.get() * 100))}%")

    # ---------------- grid (re)build ----------------
    def rebuild_grid(self):
        try:
            n = int(self.n_var.get())
            m = int(self.m_var.get())
            atomics = self.atoms_var.get().split()
            zeta = float(self.zeta_var.get())
        except ValueError:
            messagebox.showerror("Bad input", "n, m must be ints; ζ a float.")
            return
        if n < 1 or m < 1:
            messagebox.showerror("Bad input", "n and m must be >= 1.")
            return
        if not atomics or len(set(atomics)) != len(atomics):
            messagebox.showerror("Bad input", "Atomics must be unique and non-empty.")
            return

        if (n, m, atomics) != (self.n, self.m, self.atomics):
            self.initial_belief = {c: d for c, d in self.initial_belief.items()
                                   if c < n * m}
        self.n, self.m, self.atomics, self.zeta = n, m, atomics, zeta
        self.selected_cell = None

        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.cell_buttons = {}

        # Reset ALL previously-configured rows/columns. Leftover row/col
        # configs from a larger grid (e.g. going 10×10 → 6×3) keep weight=1 +
        # uniform="cells" set on indices 6..9, so they still reserve uniform
        # space and the grid stays big. Clear the old span back to weight 0 /
        # no uniform before configuring the new span.
        prev = getattr(self, "_grid_span", (0, 0))
        for c in range(max(m, prev[1])):
            self.grid_frame.columnconfigure(c, weight=0, uniform="")
        for r in range(max(n, prev[0])):
            self.grid_frame.rowconfigure(r, weight=0, uniform="")
        for c in range(m):
            self.grid_frame.columnconfigure(c, weight=1, uniform="cells")
        for r in range(n):
            self.grid_frame.rowconfigure(r, weight=1, uniform="cells")
        self._grid_span = (n, m)

        # Scale cell font / padding to the grid so big grids (e.g. 10×8) still
        # fit on screen instead of demanding hundreds of px per cell and
        # thrashing the geometry manager. The two-line "cell\nlabel %" text is
        # only shown when a cell is small enough; for dense grids we collapse
        # to a single compact line in _refresh_cell_colors via self._dense.
        big = max(n, m)
        size = 13 if big <= 6 else (11 if big <= 9 else 9)
        self._cell_font = tkfont.Font(family="DejaVu Sans", size=size,
                                      weight="bold")
        self._dense = big > 7
        pad = 5 if big <= 6 else (3 if big <= 9 else 2)

        for r in range(n):
            for c in range(m):
                cell = r * m + c
                b = tk.Button(
                    self.grid_frame, text=str(cell), font=self._cell_font,
                    width=2, height=1, relief="flat", bd=0,
                    highlightthickness=2 if self._dense else 3,
                    highlightbackground=PANEL, highlightcolor=PANEL,
                    cursor="hand2", command=lambda x=cell: self.select_cell(x))
                b.grid(row=r, column=c, padx=pad, pady=pad, sticky="nsew")
                b.bind("<Enter>", lambda e, x=cell: self._cell_hover(x, True))
                b.bind("<Leave>", lambda e, x=cell: self._cell_hover(x, False))
                self.cell_buttons[cell] = b

        self._refresh_label_list()
        self._refresh_legend()
        self._refresh_cell_colors()
        self.sel_label.config(text="No cell selected")
        self.cell_dist.delete(0, tk.END)
        self.remain_label.config(text="")
        self._draw_bar(0.0)
        self.status.config(text=f"Grid {n}×{m}  ·  atomics {atomics}  ·  "
                                f"{2**len(atomics)} labels/cell", fg=MUTED)

    def _cell_hover(self, cell, entering):
        if cell == self.selected_cell:
            return
        btn = self.cell_buttons[cell]
        base, _ = self._cell_color(cell)
        btn.config(highlightbackground=ACCENT if entering else PANEL,
                   highlightcolor=ACCENT if entering else PANEL)

    def _refresh_legend(self):
        for w in self.legend.winfo_children():
            w.destroy()
        tk.Label(self.legend, text="LEGEND", bg=PANEL, fg=MUTED,
                 font=self.font_lbl).pack(side="left", padx=(0, 10))
        def chip(color, text):
            sw = tk.Frame(self.legend, bg=color, width=16, height=16)
            sw.pack(side="left", padx=(8, 4))
            sw.pack_propagate(False)
            tk.Label(self.legend, text=text, bg=PANEL, fg=FG,
                     font=self.font_body).pack(side="left")
        for i, a in enumerate(self.atomics):
            chip(ATOM_COLORS[i % len(ATOM_COLORS)], a)
        chip(GRID_EMPTYLBL, "empty")

    def _refresh_label_list(self):
        self.label_list.delete(0, tk.END)
        self._label_for_row = []
        for lbl in all_labels(self.atomics):
            self.label_list.insert(tk.END, "  " + label_short(lbl, self.atomics))
            self._label_for_row.append(lbl)

    def _cell_color(self, cell):
        """(bg, dominant_label_tuple_or_None) for a cell."""
        if cell not in self.initial_belief:
            return GRID_EMPTY, None
        dist = self.initial_belief[cell]
        emp = empty_label(self.atomics)
        best = max(dist.items(), key=lambda kv: kv[1])
        if best[0] == emp:
            return GRID_EMPTYLBL, best
        terms = [t.strip() for t in best[0].split('&&')]
        trues = [t for t in terms if not t.startswith('!')]
        if trues and trues[0] in self.atomics:
            return ATOM_COLORS[self.atomics.index(trues[0]) % len(ATOM_COLORS)], best
        return "#7a849a", best

    def _refresh_cell_colors(self):
        for cell, btn in self.cell_buttons.items():
            color, best = self._cell_color(cell)
            fg = _ideal_text_color(color)
            if best is not None:
                short = label_short(best[0], self.atomics)
                pct = int(round(best[1] * 100))
                # Dense grids: single compact line to keep cells small.
                txt = (f"{cell} · {short}" if self._dense
                       else f"{cell}\n{short}  {pct}%")
            else:
                txt = str(cell)
            ring = SEL_RING if cell == self.selected_cell else PANEL
            btn.config(bg=color, fg=fg, text=txt, activebackground=color,
                       highlightbackground=ring, highlightcolor=ring)

    # ---------------- cell selection / editing ----------------
    def select_cell(self, cell):
        self.selected_cell = cell
        self.sel_label.config(text=f"Cell {cell}    row {cell // self.m} · "
                                   f"col {cell % self.m}")
        self._refresh_cell_dist()
        self._refresh_cell_colors()

    def _draw_bar(self, frac):
        self.bar_canvas.delete("all")
        # Use the realized width if laid out, else the requested width — do NOT
        # force update_idletasks() here: during rebuild_grid that triggers a
        # full synchronous relayout of every cell button and makes large grids
        # (e.g. 10×8) hang.
        w = self.bar_canvas.winfo_width()
        if w <= 1:
            w = self.bar_canvas.winfo_reqwidth() or 280
        frac = max(0.0, min(1.0, frac))
        col = WARN_RED if frac > 1.0 + 1e-9 else (OK_GREEN if frac >= 0.999 else ACCENT)
        self.bar_canvas.create_rectangle(0, 0, int(w * frac), 6, fill=col, width=0)
        # Redraw once the real width is known (cheap, deferred to idle).
        self.bar_canvas.after_idle(
            lambda: self._draw_bar_realwidth(frac, col))

    def _draw_bar_realwidth(self, frac, col):
        try:
            w = self.bar_canvas.winfo_width()
        except tk.TclError:
            return
        if w <= 1:
            return
        self.bar_canvas.delete("all")
        self.bar_canvas.create_rectangle(0, 0, int(w * frac), 6, fill=col, width=0)

    def _refresh_cell_dist(self):
        self.cell_dist.delete(0, tk.END)
        cell = self.selected_cell
        dist = self.initial_belief.get(cell, {})
        for lbl, p in dist.items():
            self.cell_dist.insert(
                tk.END, f" {label_short(lbl, self.atomics):<8} {p*100:5.1f}%")
        self._dist_rows = list(dist.keys())
        total = sum(dist.values())
        remain = max(0.0, 1.0 - total)
        emp = empty_label(self.atomics)
        self._draw_bar(total)
        if total > 1.0 + 1e-9:
            self.remain_label.config(
                text=f"⚠  total {total*100:.0f}% > 100% — lower or remove a label",
                fg=WARN_RED)
        else:
            msg = f"assigned {total*100:.0f}%"
            if emp not in dist and remain > 1e-9:
                msg += f"    ·    {remain*100:.0f}% → empty on save"
            self.remain_label.config(text=msg, fg=MUTED)

    def add_to_cell(self):
        if self.selected_cell is None:
            messagebox.showinfo("Pick a cell", "Click a grid cell first.")
            return
        sel = self.label_list.curselection()
        if not sel:
            messagebox.showinfo("Pick a label", "Select a label in the list.")
            return
        label = self._label_for_row[sel[0]]
        p = round(self.prob_value.get(), 4)
        if not (0.0 <= p <= 1.0):
            messagebox.showerror("Bad probability", "Probability must be 0..1.")
            return
        cell = self.selected_cell
        dist = self.initial_belief.setdefault(cell, {})
        dist[label] = p
        total = sum(dist.values())
        if total > 1.0 + 1e-9:
            messagebox.showwarning("Over 100%",
                                   f"Total at cell {cell} is {total*100:.0f}%. "
                                   "Remove or lower a label.")
        self._refresh_cell_dist()
        self._refresh_cell_colors()
        self.status.config(
            text=f"cell {cell}:  {label_short(label, self.atomics)} = {p*100:.0f}%",
            fg=OK_GREEN)

    def remove_selected(self):
        if self.selected_cell is None:
            return
        sel = self.cell_dist.curselection()
        if not sel:
            return
        label = self._dist_rows[sel[0]]
        self.initial_belief[self.selected_cell].pop(label, None)
        if not self.initial_belief[self.selected_cell]:
            self.initial_belief.pop(self.selected_cell, None)
        self._refresh_cell_dist()
        self._refresh_cell_colors()

    def clear_cell(self):
        if self.selected_cell is None:
            return
        self.initial_belief.pop(self.selected_cell, None)
        self._refresh_cell_dist()
        self._refresh_cell_colors()

    # ---------------- save ----------------
    def _finalize_belief(self):
        emp = empty_label(self.atomics)
        ib = {}
        for cell, dist in self.initial_belief.items():
            d = dict(dist)
            total = sum(d.values())
            if total > 1.0 + 1e-6:
                raise ValueError(f"cell {cell} total {total*100:.0f}% > 100%")
            if total < 1.0 - 1e-9 and emp not in d:
                d[emp] = round(1.0 - total, 6)
            ib[cell] = d
        belief = assign_probabilities_g3(
            self.n, self.m, self.atomics, initial_belief=ib, zeta=self.zeta
        )
        return ib, belief

    def _do_save(self, path):
        try:
            ib, belief = self._finalize_belief()
        except ValueError as e:
            messagebox.showerror("Invalid belief", str(e))
            return
        payload = {
            "n": self.n, "m": self.m, "atomics": self.atomics,
            "zeta": self.zeta, "initial_belief": ib, "belief": belief,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        self.status.config(text=f"✓ Saved {path}   ({len(ib)} cells with priors)",
                           fg=OK_GREEN)
        messagebox.showinfo("Saved", f"Belief written to:\n{path}")

    def save(self):
        self._do_save("belief.pkl")

    def save_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl", initialfile="belief.pkl",
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if path:
            self._do_save(path)


def main():
    root = tk.Tk()
    BeliefGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
