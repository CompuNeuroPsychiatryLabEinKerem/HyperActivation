# -*- coding: utf-8 -*-
import os, sys
import tkinter as tk
from tkinter import ttk
import easygui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import pearsonr

DIAG_ORDER  = ['CU', 'Preclinical', 'MCI', 'AD']
DIAG_COLORS = {'CU': '#2ecc71', 'Preclinical': '#3498db', 'MCI': '#9b59b6', 'AD': '#e74c3c'}
ALPHA       = 0.7

# HP model name → (lPredicted col, rPredicted col)
HP_MODELS = {
    'AABC_Predicted_LR': ('lHipPredicted',        'rHipPredicted'),
    'Potvin':            ('Potvin_Predicted_HP_L', 'Potvin_Predicted_HP_R'),
    'AABC_eTIV':         ('AABC_eTIV_Pred_L',      'AABC_eTIV_Pred_R'),
    'AABC_Brain':        ('AABC_Brain_Pred_L',      'AABC_Brain_Pred_R'),
}


# ── Backend ────────────────────────────────────────────────────────────────────

def _loadData(xlsxFile):
    runsDF    = pd.read_excel(xlsxFile, sheet_name='Runs')
    regionsDF = pd.read_excel(xlsxFile, sheet_name='Regions')
    networks  = sorted(regionsDF['Network'].dropna().unique())
    tasks     = sorted(runsDF['TaskName'].dropna().unique())
    return runsDF, regionsDF, networks, tasks


def _agg(df, aggType, axis=0):
    if aggType == 'RMS':
        return np.sqrt((df**2).mean(axis=axis))
    elif aggType == 'AverageAbs':
        return df.abs().mean(axis=axis)
    else:                           # Average
        return df.mean(axis=axis)


def _computeX(runsDF, regionsDF, seed, networks, tasks, clipZ, aggType):
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
    # Aggregate across runs first (per region), then across regions
    sbjRegion = tmp.groupby('Subject')[cols].apply(
        lambda g: _agg(g.clip(-clipZ, clipZ), aggType))
    return _agg(sbjRegion, aggType, axis=1)


def _computeY(runsDF, yMode, hpModel):
    lPred, rPred = HP_MODELS[hpModel]
    sbjDF = runsDF.drop_duplicates('Subject').dropna(subset=['Subject']).set_index('Subject')
    if yMode == 'Left':
        y = sbjDF['lHipMeasured'] / sbjDF[lPred]
    elif yMode == 'Right':
        y = sbjDF['rHipMeasured'] / sbjDF[rPred]
    elif yMode == 'Average':
        y = (sbjDF['lHipMeasured'] / sbjDF[lPred] +
             sbjDF['rHipMeasured'] / sbjDF[rPred]) / 2
    else:                                                           # Cumulative
        y = ((sbjDF['lHipMeasured'] + sbjDF['rHipMeasured']) /
             (sbjDF[lPred] + sbjDF[rPred]))
    return y, sbjDF


# ── Frontend ───────────────────────────────────────────────────────────────────

class VizApp:
    def __init__(self, root, xlsxFile=None):
        self.root = root
        self.root.title('RSP Connectivity vs Hippocampal Volume')
        self.runsDF = self.regionsDF = self.networks = self.tasks = None
        self._cbar     = None
        self._plotData = None
        self._buildUI()
        if xlsxFile:
            self._loadReport(xlsxFile)

    def _buildUI(self):
        # ── Top bar ───────────────────────────────────────────────────────
        topBar = tk.Frame(self.root)
        topBar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Button(topBar, text='Load Report…', command=self._onLoad).pack(side=tk.LEFT)
        self.fileLabel = tk.Label(topBar, text='No file loaded', anchor=tk.W, fg='gray')
        self.fileLabel.pack(side=tk.LEFT, padx=8)
        self._whoBtn = tk.Button(topBar, text='Who\'s Who', command=self._showWhoIsWho,
                                 state=tk.DISABLED)
        self._whoBtn.pack(side=tk.LEFT, padx=8)

        # ── Figure ────────────────────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.tight_layout(pad=3)
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = canvas

        # ── Control panel ─────────────────────────────────────────────────
        ctrl = tk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        # Y-axis
        yFrame = tk.LabelFrame(ctrl, text='Y-Axis (Hippocampus)', padx=4, pady=4)
        yFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N)
        self.yVar = tk.StringVar(value='Average')
        for opt in ['Left', 'Right', 'Average', 'Cumulative']:
            tk.Radiobutton(yFrame, text=opt, variable=self.yVar, value=opt,
                           command=self._update).pack(anchor=tk.W)

        # HP model
        hpFrame = tk.LabelFrame(ctrl, text='HP Model', padx=4, pady=4)
        hpFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N)
        self.hpVar     = tk.StringVar(value='AABC_Predicted_LR')
        self._hpRadios = {}
        for opt in HP_MODELS:
            rb = tk.Radiobutton(hpFrame, text=opt, variable=self.hpVar, value=opt,
                                command=self._update, state=tk.DISABLED)
            rb.pack(anchor=tk.W)
            self._hpRadios[opt] = rb

        # X-axis
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

        tk.Label(xFrame, text='Aggregation:').pack(anchor=tk.W, pady=(4, 0))
        self.aggVar = tk.StringVar(value='Average')
        for opt in ['Average', 'RMS', 'AverageAbs']:
            tk.Radiobutton(xFrame, text=opt, variable=self.aggVar, value=opt,
                           command=self._update).pack(anchor=tk.W)

        # Tasks
        taskFrame = tk.LabelFrame(ctrl, text='Tasks', padx=4, pady=4)
        taskFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N, fill=tk.Y)
        self.taskList = tk.Listbox(taskFrame, selectmode=tk.MULTIPLE,
                                   height=6, exportselection=False, width=12)
        tsb = tk.Scrollbar(taskFrame, orient=tk.VERTICAL, command=self.taskList.yview)
        self.taskList.config(yscrollcommand=tsb.set)
        self.taskList.pack(side=tk.LEFT, fill=tk.Y)
        tsb.pack(side=tk.LEFT, fill=tk.Y)
        self.taskList.bind('<<ListboxSelect>>', lambda e: self._update())

        # Networks
        netFrame = tk.LabelFrame(ctrl, text='Networks', padx=4, pady=4)
        netFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N, fill=tk.Y)
        self.netList = tk.Listbox(netFrame, selectmode=tk.MULTIPLE,
                                  height=12, exportselection=False, width=16)
        sb = tk.Scrollbar(netFrame, orient=tk.VERTICAL, command=self.netList.yview)
        self.netList.config(yscrollcommand=sb.set)
        self.netList.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.netList.bind('<<ListboxSelect>>', lambda e: self._update())

        # Diagnoses
        diagFrame = tk.LabelFrame(ctrl, text='Diagnoses', padx=4, pady=4)
        diagFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N, fill=tk.Y)
        self.diagList = tk.Listbox(diagFrame, selectmode=tk.MULTIPLE,
                                   height=len(DIAG_ORDER), exportselection=False, width=12)
        self.diagList.pack()
        for i, d in enumerate(DIAG_ORDER):
            self.diagList.insert(tk.END, d)
            self.diagList.selection_set(i)
        self.diagList.bind('<<ListboxSelect>>', lambda e: self._update())

        # Color coding
        colorFrame = tk.LabelFrame(ctrl, text='Color', padx=4, pady=4)
        colorFrame.pack(side=tk.LEFT, padx=6, anchor=tk.N)
        self.colorVar = tk.StringVar(value='Diagnosis')
        for opt in ['Diagnosis', 'Age']:
            tk.Radiobutton(colorFrame, text=opt, variable=self.colorVar, value=opt,
                           command=self._update).pack(anchor=tk.W)

    def _onLoad(self):
        path = easygui.fileopenbox(title='Select Report XLSX', filetypes=['*.xlsx'])
        if path:
            self._loadReport(path)

    def _loadReport(self, path):
        self.runsDF, self.regionsDF, self.networks, self.tasks = _loadData(path)
        self.fileLabel.config(text=os.path.basename(path), fg='black')
        self._populateLists()
        self._updateHpModelState()
        self._update()

    def _populateLists(self):
        self.taskList.delete(0, tk.END)
        self.taskList.config(height=min(len(self.tasks), 12))
        for i, t in enumerate(self.tasks):
            self.taskList.insert(tk.END, t)
            if t != 'rest':
                self.taskList.selection_set(i)

        self.netList.delete(0, tk.END)
        self.netList.config(height=min(len(self.networks), 12))
        for i, n in enumerate(self.networks):
            self.netList.insert(tk.END, n)
            self.netList.selection_set(i)

    def _updateHpModelState(self):
        cols = set(self.runsDF.columns)
        firstEnabled = None
        for name, (lCol, rCol) in HP_MODELS.items():
            available = lCol in cols and rCol in cols
            state = tk.NORMAL if available else tk.DISABLED
            self._hpRadios[name].config(state=state)
            if available and firstEnabled is None:
                firstEnabled = name
        # If current selection is disabled, switch to first available
        if self._hpRadios[self.hpVar.get()]['state'] == tk.DISABLED and firstEnabled:
            self.hpVar.set(firstEnabled)

    def _selectedNetworks(self):
        return [self.networks[i] for i in self.netList.curselection()]

    def _selectedTasks(self):
        return [self.tasks[i] for i in self.taskList.curselection()]

    def _selectedDiagnoses(self):
        return [DIAG_ORDER[i] for i in self.diagList.curselection()]

    def _showWhoIsWho(self):
        if self._plotData is None:
            return
        win = tk.Toplevel(self.root)
        win.title('Who\'s Who')
        cols = ['Subject', 'X', 'Y', 'Age', 'Diagnosis']
        tree = ttk.Treeview(win, columns=cols, show='headings')
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor=tk.CENTER)
        for _, row in self._plotData.iterrows():
            tree.insert('', tk.END, values=(
                row['Subject'],
                f'{row["X"]:.4f}',
                f'{row["Y"]:.4f}',
                f'{row["Age"]:.1f}',
                row['Diagnosis'],
            ))
        vsb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

    def _update(self):
        if self.runsDF is None:
            return
        try:
            clipZ = float(self.clipVar.get())
        except ValueError:
            return

        seed      = self.seedVar.get()
        yMode     = self.yVar.get()
        hpModel   = self.hpVar.get()
        aggType   = self.aggVar.get()
        colorMode = self.colorVar.get()
        networks  = self._selectedNetworks()
        tasks     = self._selectedTasks()
        diagnoses = self._selectedDiagnoses()

        sbjX        = _computeX(self.runsDF, self.regionsDF, seed, networks, tasks, clipZ, aggType)
        sbjY, sbjDF = _computeY(self.runsDF, yMode, hpModel)

        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self.ax.clear()

        allX, allY = [], []

        plotRows = []
        if colorMode == 'Age':
            allAges = []
            for diag in DIAG_ORDER:
                if diag not in diagnoses:
                    continue
                sbjs  = sbjDF[sbjDF['Diagnosis'] == diag].index
                xVals = sbjX.reindex(sbjs).dropna()
                yVals = sbjY.reindex(xVals.index).dropna()
                xVals = xVals.reindex(yVals.index)
                if xVals.empty:
                    continue
                ages = sbjDF.loc[xVals.index, 'Age'].values
                allX.extend(xVals.values)
                allY.extend(yVals.values)
                allAges.extend(ages)
                for sbj, x, y, age in zip(xVals.index, xVals.values, yVals.values, ages):
                    plotRows.append(dict(Subject=sbj, X=x, Y=y, Age=age, Diagnosis=diag))
            if allX:
                sc = self.ax.scatter(allX, allY, c=allAges, cmap='plasma',
                                     alpha=ALPHA, s=70, edgecolors='none',
                                     vmin=min(allAges), vmax=max(allAges))
                self._cbar = self.fig.colorbar(sc, ax=self.ax, label='Age')
        else:                                                       # Diagnosis
            for diag in DIAG_ORDER:
                if diag not in diagnoses:
                    continue
                sbjs  = sbjDF[sbjDF['Diagnosis'] == diag].index
                xVals = sbjX.reindex(sbjs).dropna()
                yVals = sbjY.reindex(xVals.index).dropna()
                xVals = xVals.reindex(yVals.index)
                if xVals.empty:
                    continue
                ages = sbjDF.loc[xVals.index, 'Age'].values
                self.ax.scatter(xVals, yVals, label=diag, color=DIAG_COLORS[diag],
                                alpha=ALPHA, s=70, edgecolors='none')
                allX.extend(xVals.values)
                allY.extend(yVals.values)
                for sbj, x, y, age in zip(xVals.index, xVals.values, yVals.values, ages):
                    plotRows.append(dict(Subject=sbj, X=x, Y=y, Age=age, Diagnosis=diag))
            self.ax.legend(loc='best', framealpha=0.7)

        self._plotData = pd.DataFrame(plotRows).sort_values('Subject').reset_index(drop=True) \
                         if plotRows else None
        self._whoBtn.config(state=tk.NORMAL if self._plotData is not None else tk.DISABLED)

        if len(allX) >= 3:
            r, p = pearsonr(allX, allY)
            pStr = f'p={p:.3f}' if p >= 0.001 else f'p={p:.2e}'
            self.ax.annotate(f'r={r:.3f}  {pStr}', xy=(0.05, 0.95),
                             xycoords='axes fraction', va='top',
                             fontsize=10, fontstyle='italic')

        seedTag = 'RspAvg' if seed == 'Avg' else 'RspSum'
        self.ax.set_xlabel(f'{aggType} z-corr ({seedTag}, clip±{clipZ})')
        self.ax.set_ylabel(f'Norm. Hippocampus ({yMode}, {hpModel})')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()


def run(xlsxFile=None):
    root = tk.Tk()
    VizApp(root, xlsxFile)
    root.mainloop()


if __name__ == '__main__':
    xlsxFile = sys.argv[1] if len(sys.argv) == 2 else None
    run(xlsxFile)
