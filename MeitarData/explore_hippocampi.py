import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

df = pd.read_csv("hippocampi_estimates.csv")

df["LeftRatio"]  = df["lHipMeasured"] / df["lHipPredicted"]
df["RightRatio"] = df["rHipMeasured"] / df["rHipPredicted"]
df["TotalRatio"] = (df["lHipMeasured"] + df["rHipMeasured"]) / (df["lHipPredicted"] + df["rHipPredicted"])
df["minRatio"]   = df[["LeftRatio", "RightRatio"]].min(axis=1)
df["maxRatio"]   = df[["LeftRatio", "RightRatio"]].max(axis=1)

MEASURES   = ["TotalRatio", "LeftRatio", "RightRatio", "minRatio", "maxRatio"]
DIAG_ORDER = ["CU", "Preclinical", "MCI", "AD"]
DIAG_COLOR = {"CU": "#2196F3", "Preclinical": "#4CAF50", "MCI": "#FF9800", "AD": "#E53935"}


class HippocampiExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Hippocampi Volume Ratios")

        # ── controls ────────────────────────────────────────────────────────
        ctrl = ttk.Frame(root, padding=(8, 6))
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="X:").pack(side=tk.LEFT)
        self.x_var = tk.StringVar(value="minRatio")
        x_cb = ttk.Combobox(ctrl, textvariable=self.x_var, values=MEASURES,
                             state="readonly", width=13)
        x_cb.pack(side=tk.LEFT, padx=(2, 16))
        x_cb.bind("<<ComboboxSelected>>", self._refresh)

        ttk.Label(ctrl, text="Y:").pack(side=tk.LEFT)
        self.y_var = tk.StringVar(value="maxRatio")
        y_cb = ttk.Combobox(ctrl, textvariable=self.y_var, values=MEASURES,
                             state="readonly", width=13)
        y_cb.pack(side=tk.LEFT, padx=(2, 16))
        y_cb.bind("<<ComboboxSelected>>", self._refresh)

        self.log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="log(ratio)", variable=self.log_var,
                        command=self._refresh).pack(side=tk.LEFT, padx=8)

        # ── figure ──────────────────────────────────────────────────────────
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        self._scatter_map = []  # list of (PathCollection, sub-DataFrame)
        self._annot = None
        self._refresh()

    # ── plotting ─────────────────────────────────────────────────────────────
    def _values(self, col):
        v = df[col].values.copy()
        if self.log_var.get():
            v = np.log(v)
        return v

    def _refresh(self, _event=None):
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        use_log = self.log_var.get()
        label = "log(ratio)" if use_log else "ratio"

        self.ax.cla()
        self._scatter_map = []

        for diag in DIAG_ORDER:
            sub = df[df["Diagnosis"] == diag]
            if sub.empty:
                continue
            xv = sub[x_col].values.copy()
            yv = sub[y_col].values.copy()
            if use_log:
                xv, yv = np.log(xv), np.log(yv)
            sc = self.ax.scatter(
                xv, yv,
                c=DIAG_COLOR.get(diag, "gray"),
                label=diag, s=80, alpha=0.88,
                edgecolors="white", linewidths=0.6, zorder=3,
            )
            self._scatter_map.append((sc, sub, xv, yv))

        # reference line: y = x (makes sense for any pair of ratios)
        all_x = np.concatenate([xv for _, _, xv, _ in self._scatter_map])
        all_y = np.concatenate([yv for _, _, _, yv in self._scatter_map])
        lo = min(all_x.min(), all_y.min())
        hi = max(all_x.max(), all_y.max())
        pad = (hi - lo) * 0.04
        ref = [lo - pad, hi + pad]
        self.ax.plot(ref, ref, "k--", lw=1, alpha=0.3, zorder=1)

        # horizontal/vertical reference at 0 (log) or 1 (ratio)
        ref_val = 0.0 if use_log else 1.0
        self.ax.axhline(ref_val, color="gray", lw=0.8, alpha=0.4, zorder=1)
        self.ax.axvline(ref_val, color="gray", lw=0.8, alpha=0.4, zorder=1)

        xlabel = f"log({x_col})" if use_log else x_col
        ylabel = f"log({y_col})" if use_log else y_col
        self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)
        self.ax.set_title("Hippocampi  measured / predicted", fontsize=12)
        self.ax.legend(title="Diagnosis", framealpha=0.9, fontsize=9)
        self.ax.grid(True, alpha=0.25)

        self._annot = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="gray"),
            fontsize=9, zorder=10,
        )
        self._annot.set_visible(False)
        self.canvas.draw()

    # ── hover tooltip ────────────────────────────────────────────────────────
    def _on_hover(self, event):
        if event.inaxes != self.ax or self._annot is None:
            return
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        use_log = self.log_var.get()

        hit = False
        for sc, sub, xv, yv in self._scatter_map:
            cont, ind = sc.contains(event)
            if cont:
                i = ind["ind"][0]
                row = sub.iloc[i]
                px, py = xv[i], yv[i]
                self._annot.xy = (px, py)
                xr = row[x_col]
                yr = row[y_col]
                fmt = ".3f"
                xdisp = f"log={px:{fmt}}" if use_log else f"{xr:{fmt}}"
                ydisp = f"log={py:{fmt}}" if use_log else f"{yr:{fmt}}"
                self._annot.set_text(
                    f"{row['SubjectID']}  ({row['Diagnosis']}, {row['Sex']}, {row['Age']}y)\n"
                    f"{x_col}: {xdisp}\n"
                    f"{y_col}: {ydisp}"
                )
                self._annot.set_visible(True)
                hit = True
                break

        if not hit:
            self._annot.set_visible(False)
        self.canvas.draw_idle()


root = tk.Tk()
root.minsize(680, 520)
HippocampiExplorer(root)
root.mainloop()
