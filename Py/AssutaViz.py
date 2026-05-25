# -*- coding: utf-8 -*-
import sys
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import pearsonr

DIAG_ORDER  = ['CU', 'Preclinical', 'MCI', 'AD']
DIAG_COLORS = {'CU': '#2ecc71', 'Preclinical': '#3498db', 'MCI': '#9b59b6', 'AD': '#e74c3c'}
ALPHA       = 0.7


# ── Backend ────────────────────────────────────────────────────────────────────

def _loadData(xlsxFile):
    runsDF    = pd.read_excel(xlsxFile, sheet_name='Runs')
    regionsDF = pd.read_excel(xlsxFile, sheet_name='Regions')
    networks  = sorted(regionsDF['Network'].dropna().unique())
    tasks     = sorted(runsDF['TaskName'].dropna().unique())
    return runsDF, regionsDF, networks, tasks


def _computeX(runsDF, regionsDF, seed, networks, tasks, clipZ):
    seedTag = 'RspAvg' if seed == 'Avg' else 'RspSum'
    mask    = regionsDF['Network'].isin(networks) if networks else pd.Series(True, index=regionsDF.index)
    labels  = regionsDF[mask]['Label'].values
    cols    = [f'zCorr_{seedTag}_Reg{lbl}' for lbl in labels
               if f'zCorr_{seedTag}_Reg{lbl}' in runsDF.columns]
    if not cols:
        return pd.Series(dtype=float)
    tmp = runsDF[['Subject', 'TaskName'] + cols].copy()
    if tasks:
        tmp = tmp[tmp['TaskName'].isin(tasks)]
    tmp = tmp.dropna(subset=['Subject'])
    # Average across runs first (per region), then across regions
    sbjRegion = tmp.groupby('Subject')[cols].apply(lambda g: g.clip(-clipZ, clipZ).mean())
    return sbjRegion.mean(axis=1)


def _computeY(runsDF, yMode):
    sbjDF = runsDF.drop_duplicates('Subject').dropna(subset=['Subject']).set_index('Subject')
    if yMode == 'Left':
        y = sbjDF['lHipMeasured'] / sbjDF['lHipPredicted']
    elif yMode == 'Right':
        y = sbjDF['rHipMeasured'] / sbjDF['rHipPredicted']
    elif yMode == 'Average':
        y = (sbjDF['lHipMeasured'] / sbjDF['lHipPredicted'] +
             sbjDF['rHipMeasured'] / sbjDF['rHipPredicted']) / 2
    else:                                                           # Cumulative
        y = ((sbjDF['lHipMeasured'] + sbjDF['rHipMeasured']) /
             (sbjDF['lHipPredicted'] + sbjDF['rHipPredicted']))
    return y, sbjDF


# ── Frontend ───────────────────────────────────────────────────────────────────

class VizApp:
    def __init__(self, root, xlsxFile):
        self.root = root
        self.root.title('RSP Connectivity vs Hippocampal Volume')
        self.runsDF, self.regionsDF, self.networks, self.tasks = _loadData(xlsxFile)
        self._buildUI()
        self._update()

    def _buildUI(self):
        # Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.tight_layout(pad=3)
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = canvas

        # Control panel
        ctrl = tk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        # ── Y-axis ────────────────────────────────────────────────────────
        yFrame = tk.LabelFrame(ctrl, text='Y-Axis (Hippocampus)', padx=4, pady=4)
        yFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N)
        self.yVar = tk.StringVar(value='Average')
        for opt in ['Left', 'Right', 'Average', 'Cumulative']:
            tk.Radiobutton(yFrame, text=opt, variable=self.yVar, value=opt,
                           command=self._update).pack(anchor=tk.W)

        # ── X-axis ────────────────────────────────────────────────────────
        xFrame = tk.LabelFrame(ctrl, text='X-Axis (Connectivity)', padx=4, pady=4)
        xFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N)

        clipRow = tk.Frame(xFrame)
        clipRow.pack(anchor=tk.W, pady=(0, 4))
        tk.Label(clipRow, text='ClipZ:').pack(side=tk.LEFT)
        self.clipVar = tk.StringVar(value='3.0')
        clipEntry = tk.Entry(clipRow, textvariable=self.clipVar, width=5)
        clipEntry.pack(side=tk.LEFT, padx=4)
        clipEntry.bind('<Return>', lambda e: self._update())

        tk.Label(xFrame, text='Seed:').pack(anchor=tk.W)
        self.seedVar = tk.StringVar(value='Avg')
        for opt in ['Avg', 'Sum']:
            tk.Radiobutton(xFrame, text=opt, variable=self.seedVar, value=opt,
                           command=self._update).pack(anchor=tk.W)

        # ── Tasks ─────────────────────────────────────────────────────────
        taskFrame = tk.LabelFrame(ctrl, text='Tasks', padx=4, pady=4)
        taskFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N, fill=tk.Y)

        self.taskList = tk.Listbox(taskFrame, selectmode=tk.MULTIPLE,
                                   height=min(len(self.tasks), 12),
                                   exportselection=False, width=12)
        tsb = tk.Scrollbar(taskFrame, orient=tk.VERTICAL, command=self.taskList.yview)
        self.taskList.config(yscrollcommand=tsb.set)
        self.taskList.pack(side=tk.LEFT, fill=tk.Y)
        tsb.pack(side=tk.LEFT, fill=tk.Y)

        for i, t in enumerate(self.tasks):
            self.taskList.insert(tk.END, t)
            if t != 'rest':
                self.taskList.selection_set(i)      # all except rest selected by default

        self.taskList.bind('<<ListboxSelect>>', lambda e: self._update())

        # ── Networks ──────────────────────────────────────────────────────
        netFrame = tk.LabelFrame(ctrl, text='Networks', padx=4, pady=4)
        netFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N, fill=tk.Y)

        self.netList = tk.Listbox(netFrame, selectmode=tk.MULTIPLE,
                                  height=min(len(self.networks), 12),
                                  exportselection=False, width=16)
        sb = tk.Scrollbar(netFrame, orient=tk.VERTICAL, command=self.netList.yview)
        self.netList.config(yscrollcommand=sb.set)
        self.netList.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        for i, n in enumerate(self.networks):
            self.netList.insert(tk.END, n)
            self.netList.selection_set(i)           # ALL selected by default

        self.netList.bind('<<ListboxSelect>>', lambda e: self._update())

    def _selectedNetworks(self):
        return [self.networks[i] for i in self.netList.curselection()]

    def _selectedTasks(self):
        return [self.tasks[i] for i in self.taskList.curselection()]

    def _update(self):
        try:
            clipZ = float(self.clipVar.get())
        except ValueError:
            return

        seed     = self.seedVar.get()
        yMode    = self.yVar.get()
        networks = self._selectedNetworks()
        tasks    = self._selectedTasks()

        sbjX        = _computeX(self.runsDF, self.regionsDF, seed, networks, tasks, clipZ)
        sbjY, sbjDF = _computeY(self.runsDF, yMode)

        self.ax.clear()
        allX, allY = [], []
        for diag in DIAG_ORDER:
            sbjs  = sbjDF[sbjDF['Diagnosis'] == diag].index
            xVals = sbjX.reindex(sbjs).dropna()
            yVals = sbjY.reindex(xVals.index).dropna()
            xVals = xVals.reindex(yVals.index)
            if xVals.empty:
                continue
            self.ax.scatter(xVals, yVals, label=diag, color=DIAG_COLORS[diag],
                            alpha=ALPHA, s=70, edgecolors='none')
            allX.extend(xVals.values)
            allY.extend(yVals.values)

        if len(allX) >= 3:
            r, p = pearsonr(allX, allY)
            pStr = f'p={p:.3f}' if p >= 0.001 else f'p={p:.2e}'
            self.ax.annotate(f'r={r:.3f}  {pStr}', xy=(0.05, 0.95),
                             xycoords='axes fraction', va='top',
                             fontsize=10, fontstyle='italic')

        seedTag = 'RspAvg' if seed == 'Avg' else 'RspSum'
        self.ax.set_xlabel(f'Mean z-corr ({seedTag}, clip±{clipZ})')
        self.ax.set_ylabel(f'Norm. Hippocampus ({yMode})')
        self.ax.legend(loc='best', framealpha=0.7)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()


def run(xlsxFile):
    root = tk.Tk()
    VizApp(root, xlsxFile)
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python AssutaViz.py <report.xlsx>')
        sys.exit(1)
    run(sys.argv[1])
