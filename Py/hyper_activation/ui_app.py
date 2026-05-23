"""Tkinter-based desktop UI for HyperActivation exploration."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .data_access import DataRepository
from .measurements import AXES_NAMES, MeasurementService
from .models import SelectionState
from .stats_utils import calc_pvalues, calc_partial_correlations, format_stats_text, make_color


class HyperActivationApp:
    """Desktop UI that replaces the previous ipywidgets experience."""

    def __init__(self, repository: DataRepository | None = None) -> None:
        self.repo = repository or DataRepository()
        self.measurements = MeasurementService(self.repo)

        self.root = tk.Tk()
        self.root.title("HyperActivation Explorer")

        self.axis_vars = {
            "x": tk.StringVar(value=AXES_NAMES[0]),
            "y": tk.StringVar(value=AXES_NAMES[1]),
            "c": tk.StringVar(value=AXES_NAMES[-1]),
        }
        self.c_axis_visible = tk.BooleanVar(value=True)

        self.domain = tk.StringVar(value="Space")
        self.regress_age = tk.StringVar(value="none")
        self.penalty = tk.StringVar(value="none")
        self.transformation = tk.StringVar(value="score")  # "score", "log(score)", "1/score"

        self.rsp_vertices = tk.StringVar(value="Full")
        self.rsp_source = tk.StringVar(value="Default")
        self.hpc_size = tk.StringVar(value="200")
        self.conn_units = tk.StringVar(value="Singles")
        self.conn_stat = tk.StringVar(value="mean")
        self.conn_threshold = tk.DoubleVar(value=0.3)

        self.hp_source = tk.StringVar(value="FS")
        self.hp_stat = tk.StringVar(value="mean")

        self.color_margin = tk.DoubleVar(value=1.3)

        self.keyword_listbox: tk.Listbox | None = None
        self.stats_var = tk.StringVar(value="Press Go! to compute statistics.")

        self.num_permutations = tk.IntVar(value=1000)
        self.use_loo = tk.BooleanVar(value=True)
        self.reference_value = tk.StringVar(value="median")
        self.enable_classification = tk.BooleanVar(value=True)
        self.remove_rm = tk.BooleanVar(value=False)
        self.remove_ad = tk.BooleanVar(value=False)

        self.figure = Figure(figsize=(6, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas: FigureCanvasTkAgg | None = None
        
        # Partial correlation matrix figure
        self.corr_figure = Figure(figsize=(3, 3))
        self.corr_ax = self.corr_figure.add_subplot(111)
        self.corr_canvas: FigureCanvasTkAgg | None = None

        self._build_layout()
        self._draw_placeholder()

    # ------------------------------------------------------------------ UI setup
    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="ns")
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self._build_plot(plot_frame)
        self._build_axis_controls(controls_frame)
        self._build_correlation_matrix(controls_frame)
        self._build_options(options_frame)

    def _build_plot(self, frame: ttk.Frame) -> None:
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        stats_label = ttk.Label(frame, textvariable=self.stats_var, justify=tk.LEFT)
        stats_label.pack(fill=tk.X, pady=(8, 0))

    def _build_axis_controls(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Axes").grid(row=0, column=0, columnspan=2, pady=(0, 4))
        for idx, axis in enumerate(("x", "y", "c")):
            ttk.Label(frame, text=f"{axis.upper()}-Axis").grid(row=idx + 1, column=0, sticky="w")
            var = self.axis_vars[axis]
            menu = ttk.OptionMenu(frame, var, var.get(), *AXES_NAMES)
            menu.grid(row=idx + 1, column=1, sticky="ew", padx=(4, 0))

        ttk.Checkbutton(
            frame, text="Show C-Axis", variable=self.c_axis_visible, command=self._on_axis_toggle
        ).grid(row=4, column=0, columnspan=2, pady=(8, 0))

        go_button = ttk.Button(frame, text="Go!", command=self._on_go)
        go_button.grid(row=5, column=0, columnspan=2, pady=(12, 0), sticky="ew")

        frame.columnconfigure(1, weight=1)

    def _build_correlation_matrix(self, frame: ttk.Frame) -> None:
        """Build partial correlation matrix display in side panel."""
        ttk.Label(frame, text="Partial Correlations").grid(row=6, column=0, columnspan=2, pady=(12, 4))
        
        self.corr_canvas = FigureCanvasTkAgg(self.corr_figure, master=frame)
        self.corr_canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._draw_empty_corr_matrix()

    def _draw_empty_corr_matrix(self) -> None:
        """Draw empty correlation matrix placeholder."""
        self.corr_ax.clear()
        self.corr_ax.text(0.5, 0.5, "Press Go! to\ncalculate", 
                         ha="center", va="center", fontsize=10)
        self.corr_ax.set_xticks([])
        self.corr_ax.set_yticks([])
        self.corr_ax.set_xlim(0, 1)
        self.corr_ax.set_ylim(0, 1)
        self.corr_figure.tight_layout()
        self.corr_canvas.draw_idle()

    def _draw_correlation_matrix(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, xlabel: str, ylabel: str, zlabel: str) -> None:
        """Draw partial correlation matrix heatmap."""
        self.corr_ax.clear()
        
        try:
            corr_matrix = calc_partial_correlations(x, y, z, (xlabel, ylabel, zlabel))
            
            # Create heatmap
            im = self.corr_ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            
            # Set labels
            labels = [xlabel[:8], ylabel[:8], zlabel[:8]]  # Truncate long labels
            self.corr_ax.set_xticks([0, 1, 2])
            self.corr_ax.set_yticks([0, 1, 2])
            self.corr_ax.set_xticklabels(labels, rotation=45, ha='right')
            self.corr_ax.set_yticklabels(labels)
            
            # Add text annotations with correlation values
            for i in range(3):
                for j in range(3):
                    val = corr_matrix[i, j]
                    text = f"{val:.2f}" if np.isfinite(val) else "nan"
                    color = 'white' if abs(val) > 0.5 else 'black'
                    self.corr_ax.text(j, i, text, ha="center", va="center", 
                                     color=color, fontsize=8, fontweight='bold')
            
            # Add colorbar
            self.corr_figure.colorbar(im, ax=self.corr_ax, fraction=0.046, pad=0.04)
            
        except Exception as e:
            self.corr_ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}", 
                             ha="center", va="center", fontsize=8)
        
        self.corr_figure.tight_layout()
        self.corr_canvas.draw_idle()

    def _build_options(self, frame: ttk.Frame) -> None:
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        self._build_orientation_frame(frame)
        self._build_rsp_frame(frame)
        self._build_hpc_frame(frame)
        self._build_connectivity_frame(frame)
        self._build_hp_frame(frame)
        self._build_misc_frame(frame)
        self._build_color_frame(frame)
        self._build_stats_frame(frame)

    def _build_orientation_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Orientation")
        box.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(box, text="Domain").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(box, self.domain, self.domain.get(), "Space", "Time", "Person").grid(
            row=0, column=1, sticky="ew", padx=4
        )

        ttk.Label(box, text="Regress Age").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(box, self.regress_age, self.regress_age.get(), "none", "before", "after").grid(
            row=1, column=1, sticky="ew", padx=4
        )

        ttk.Label(box, text="Transform").grid(row=2, column=0, sticky="w")
        ttk.OptionMenu(
            box, self.transformation, self.transformation.get(), "score", "log(score)", "1/score"
        ).grid(row=2, column=1, sticky="ew", padx=4)

        ttk.Label(box, text="Penalty").grid(row=3, column=0, sticky="w")
        ttk.OptionMenu(
            box, self.penalty, self.penalty.get(), "none", "full", "sqrt", "fullinv", "sqrtinv"
        ).grid(row=3, column=1, sticky="ew", padx=4)

        box.columnconfigure(1, weight=1)

    def _build_hpc_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="HPC Activation")
        box.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        ttk.Label(box, text="Size").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(box, self.hpc_size, self.hpc_size.get(), "200", "100", "50").grid(
            row=0, column=1, sticky="ew", padx=4
        )
        box.columnconfigure(1, weight=1)

    def _build_rsp_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="RSP Activation")
        box.grid(row=0, column=1, sticky="ew")

        ttk.Label(box, text="# Vertices").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(box, self.rsp_vertices, self.rsp_vertices.get(), "Full", "100", "50", "20").grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Label(box, text="Source").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(box, self.rsp_source, self.rsp_source.get(), "Default", "Beta", "PSC", "Z").grid(
            row=1, column=1, sticky="ew", padx=4
        )
        box.columnconfigure(1, weight=1)

    def _build_connectivity_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Connectivity")
        box.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(8, 0))

        ttk.Label(box, text="Units").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(box, self.conn_units, self.conn_units.get(), "Singles", "Bilaterals", "Networks").grid(
            row=0, column=1, sticky="ew", padx=4
        )

        ttk.Label(box, text="Stat").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(box, self.conn_stat, self.conn_stat.get(), "mean", "count", "z").grid(
            row=1, column=1, sticky="ew", padx=4
        )

        ttk.Label(box, text="Threshold").grid(row=2, column=0, sticky="w")
        ttk.Scale(box, variable=self.conn_threshold, from_=0.16, to=0.5, orient=tk.HORIZONTAL).grid(
            row=2, column=1, sticky="ew", padx=4
        )

        ttk.Label(box, text="Terms").grid(row=3, column=0, sticky="nw")
        self.keyword_listbox = tk.Listbox(box, selectmode=tk.MULTIPLE, height=4, exportselection=False)
        for term in self.repo.get_connectivity_terms():
            self.keyword_listbox.insert(tk.END, term)
        self.keyword_listbox.grid(row=3, column=1, sticky="ew", padx=4, pady=(4, 0))

        box.columnconfigure(1, weight=1)

    def _build_hp_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Hippocampal Volume")
        box.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(box, text="Source").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(box, self.hp_source, self.hp_source.get(), "FS", "CFT", "CFT_MNI").grid(
            row=0, column=1, sticky="ew", padx=4
        )

        ttk.Label(box, text="Stat").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(box, self.hp_stat, self.hp_stat.get(), "mean", "left", "right", "min", "max").grid(
            row=1, column=1, sticky="ew", padx=4
        )

        box.columnconfigure(1, weight=1)

    def _build_misc_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Misc")
        box.grid(row=1, column=2, sticky="ew", pady=(8, 0), padx=(8, 0))

        ttk.Checkbutton(box, text="Remove RM", variable=self.remove_rm).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(box, text="Remove AD", variable=self.remove_ad).grid(
            row=1, column=0, sticky="w"
        )

    def _build_color_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Color Properties")
        box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        ttk.Label(box, text="Margin").grid(row=0, column=0, sticky="w")
        ttk.Scale(box, variable=self.color_margin, from_=0.5, to=2.0, orient=tk.HORIZONTAL).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        box.columnconfigure(1, weight=1)

    def _build_stats_frame(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Statistics")
        box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        ttk.Label(box, text="# Permutations").grid(row=0, column=0, sticky="w")
        perm_spin = tk.Spinbox(
            box, from_=10, to=10000, increment=10, textvariable=self.num_permutations, width=8
        )
        perm_spin.grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Checkbutton(box, text="Use Leave-One-Out", variable=self.use_loo).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )

        ttk.Label(box, text="Reference Value").grid(row=2, column=0, sticky="w")
        ttk.OptionMenu(box, self.reference_value, self.reference_value.get(), "mean", "median").grid(
            row=2, column=1, sticky="ew", padx=4
        )

        ttk.Checkbutton(box, text="Enable Classification", variable=self.enable_classification).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        box.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ Helpers
    def _draw_placeholder(self) -> None:
        self.ax.clear()
        self.ax.set_title("HyperActivation Explorer")
        self.ax.set_xlabel("X-Axis")
        self.ax.set_ylabel("Y-Axis")
        self.canvas.draw_idle()

    def _collect_keywords(self) -> Tuple[str, ...]:
        if not self.keyword_listbox:
            return ()
        indices = self.keyword_listbox.curselection()
        return tuple(self.keyword_listbox.get(i) for i in indices)

    def _collect_state(self) -> SelectionState:
        return SelectionState.from_controls(
            domain=self.domain.get(),
            rsp_vertices=self.rsp_vertices.get(),
            rsp_source=self.rsp_source.get(),
            hpc_size=self.hpc_size.get(),
            conn_units=self.conn_units.get(),
            conn_stat=self.conn_stat.get(),
            conn_threshold=float(self.conn_threshold.get()),
            conn_keywords=self._collect_keywords(),
            hp_source=self.hp_source.get(),
            hp_stat=self.hp_stat.get(),
            penalty=self.penalty.get(),
            transformation=self.transformation.get(),
            regress_age=self.regress_age.get(),
            color_margin=float(self.color_margin.get()),
            c_axis_visible=bool(self.c_axis_visible.get()),
            num_permutations=int(self.num_permutations.get()),
            use_loo=bool(self.use_loo.get()),
            reference_value=self.reference_value.get(),
            enable_classification=bool(self.enable_classification.get()),
            remove_rm=bool(self.remove_rm.get()),
            remove_ad=bool(self.remove_ad.get()),
        )

    def _read_basic_data(self, selections: SelectionState):
        x_series = self.measurements.series_for_modality(self.axis_vars["x"].get(), selections)
        y_series = self.measurements.series_for_modality(self.axis_vars["y"].get(), selections)

        shared_idx = x_series.index.intersection(y_series.index)
        x = x_series.loc[shared_idx].values
        y = y_series.loc[shared_idx].values

        if selections.c_axis_visible:
            z_series = self.measurements.series_for_modality(self.axis_vars["c"].get(), selections)
            z = z_series.loc[shared_idx].values
        else:
            z = np.zeros_like(x)

        labels = [self.axis_vars["x"].get(), self.axis_vars["y"].get()]
        return x, y, z, labels

    def _on_axis_toggle(self) -> None:
        if not self.c_axis_visible.get():
            self.stats_var.set("C-axis hidden; colors will default to neutral.")

    def _on_go(self) -> None:
        try:
            selections = self._collect_state()
            x, y, z, labels = self._read_basic_data(selections)
            colors = make_color(z, margin=selections.color_margin)
            self._draw_plot(x, y, labels, colors)
            
            # Update correlation matrix
            if selections.c_axis_visible:
                zlabel = self.axis_vars["c"].get()
            else:
                zlabel = "Z"
            self._draw_correlation_matrix(x, y, z, labels[0], labels[1], zlabel)
            
            pvalues = calc_pvalues(
                x,
                y,
                z,
                num_permutations=selections.num_permutations,
                use_loo=selections.use_loo,
                reference_value=selections.reference_value,
                enable_classification=self.enable_classification.get(),
            )
            self.stats_var.set(format_stats_text(pvalues))
        except Exception as exc:
            messagebox.showerror("HyperActivation", str(exc))

    def _draw_plot(self, x, y, labels, colors) -> None:
        self.ax.clear()
        self.ax.scatter(x, y, c=colors)
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ Public
    def run(self) -> None:
        self.root.mainloop()


def launch_app(project_dir: str | None = None) -> None:
    """Convenience wrapper that instantiates and starts the UI."""
    repo = DataRepository(project_dir=project_dir) if project_dir else None
    app = HyperActivationApp(repo)
    app.run()

