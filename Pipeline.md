# HyperActivation Pipeline: Raw dtseries → Scatter Plot

## Overview

```
fMRIPrep output
    │
    ├── *_Atlas_s0.dtseries.nii   (BOLD, 91K grayordinates)
    └── *_desc-confounds_*.tsv    (motion, GSR, CompCor, etc.)
         │
         ▼
[1] AssutaCleanBold.py  ──── CleanCfg.yaml
         │
         └── *_Atlas_s0_cleaned.dtseries.nii
              │
              ▼
[2] AssutaParcellate.py  ──── dlabel file + label text file + Runs2Tasks.txt
              │
              └── All_114_Timecourses.pkl   (HPC-generated, pandas 3.x)
                   │
                   ▼
         [3] ConvertPkl.py  (one-time, run in conv_env with pandas 3.x)
                   │
                   ├── All_114_Arrays.pkl   (numpy only)
                   └── All_114_Meta.xlsx    (Runs + Labels sheets)
                        │
                        ▼
              [4] AssutaFC.py
                        │
                        └── All_114_FC.pkl
                             │
                             ▼
                   [5] AssutaExport.py  ──── hippocampi_estimates.csv
                             │
                             └── All_114_Report.xlsx
                                  │
                                  ▼
                        [6] AssutaViz.py  (interactive)
```

---

## Stage 1 — BOLD Cleaning (`AssutaCleanBold.py`)

**Runs on:** HPC (SLURM via `SimpleGateScript.sh`)

**Input:**
- `bold_dir/`: `*_Atlas_s0[_fwhmN].dtseries.nii` — cleaned CIFTI BOLD files from fMRIPrep
- `confounds_dir/`: `*_desc-confounds_timeseries.tsv` — fMRIPrep confound TSVs
- `CleanCfg.yaml` — configuration file (see options below)

**Output:**
- `dest_dir/`: `*_Atlas_s0_cleaned.dtseries.nii` — one cleaned file per input run

**Configuration (`CleanCfg.yaml`):**

```yaml
PATHS:
  root_dir:      /path/to/project          # all relative paths resolved from here
  bold_dir:      DtSeries                  # input BOLD files
  confounds_dir: Confounds                 # input confound TSVs
  dest_dir:      DtSeriesClean             # output cleaned files
  keyword:       "*"                       # task keyword filter (* = all tasks)
  fwhm:          0                         # surface smoothing (0 = none)

MISC:
  TR:            2.02                      # repetition time (seconds)
  remove1stVols: 2                         # volumes removed from start of each run

SCRUB:
  Do:            false                     # enable scrubbing
  fdThr:         -1                        # framewise displacement threshold (-1 = off)
  dvarsThr:      75                        # DVARS threshold (-1 = off)
  minStreakThr:  -1                        # min inlier streak to keep (-1 = off)
  acceptThr:     -1                        # min fraction of inliers to accept run (-1 = off)

REGRESS:
  Do:            true                      # enable confound regression
  useConfounds:                            # list of column names or regexes
    - global_signal
    - framewise_displacement
    - csf(_derivative)?
    - motion_outlier.+
    - (rot|trans)_[xyz](_derivative1)?

FILTER:
  Do:            true                      # enable bandpass filtering
  iirOrder:      3                         # IIR filter order
  band:          [0.008, 0.25]             # [low, high] Hz
```

**Notes:**
- File pairing is done by signature: `sub-X_ses-Y_task-Z_run-N` (split on `_Atlas`)
- TR mismatch between BOLD and confounds TSV → file is skipped with a log message
- If a confound name is not found in the TSV → **silently skipped** (known limitation)
- Parallelized via `ProcessPoolExecutor`; on HPC uses `SLURM_CPUS_PER_TASK`

**How to run (HPC):**
```bash
sbatch --export=PY_SCRIPT="Py/AssutaCleanBold.py",PY_ARGS="Py/CleanCfg.yaml" Py/SimpleGateScript.sh
```

---

## Stage 2 — Parcellation (`AssutaParcellate.py`)

**Runs on:** HPC (SLURM via `SimpleGateScript.sh`)

**Input:**
- `dtseriesDir/`: `*_Atlas_s0_cleaned.dtseries.nii` — output of Stage 1
- `runs2tasksFile`: `Runs2Tasks.txt` — maps cleaned task runs to subject/run/task identity
- `labelFile`: Yeo label text file (columns: `parcelIdx desc r g b a`)
- `dlabelFile`: Yeo 17-network 91K dlabel CIFTI (default: `Yeo2011_17Networks_91K.split_components.dlabel.nii`)

**`Runs2Tasks.txt` format:**
```
{trueSbj}-{fullRunIdx}-{taskName}__{boldFileSignature}_
# e.g.: AvSh-09-mental__sub-AvShB_ses-001_task-mental_run-003_
```
Rest runs (not listed) are auto-assigned: session suffix A → Run 1, B → Run 2.

**Output:** `All_114_Timecourses.pkl` containing:

| Key           | Type                  | Description                                      |
|---------------|-----------------------|--------------------------------------------------|
| `Timecourses` | list of `(114, nTP)` arrays | Parcel sum timeseries per run             |
| `parcelsSize` | `(114,)` array        | Number of vertices per parcel                    |
| `Runs`        | DataFrame             | `Subject, Run, TaskName, OrigID` — one row/run   |
| `Labels`      | DataFrame             | `Label, Hemi, Network, SubParcel, Desc` per parcel |

**Preprocessing per run (in `parcellateFile`):**
1. Remove per-vertex temporal mean
2. Normalize by global RMS
3. Sum across vertices within each parcel → `(114, nTP)` timeseries

**How to run (HPC):**
```bash
sbatch --export=PY_SCRIPT="Py/AssutaParcellate.py",PY_ARGS="<dtseriesDir> <runs2tasksFile> <labelFile> <outputFile>" Py/SimpleGateScript.sh
```

---

## Stage 3 — Pickle Conversion (`ConvertPkl.py`)

**One-time step.** Required because the HPC (fMRIPrep 24.x era) runs pandas 3.x, while the local analysis environment uses pandas 2.2.x — making the HPC-generated pickle unreadable locally.

**Runs in:** `Py/conv_env/` (dedicated venv with pandas ≥ 3.0)

**Input:** `All_114_Timecourses.pkl`

**Output:**
- `All_114_Arrays.pkl` — numpy-only: `{Timecourses, parcelsSize}`
- `All_114_Meta.xlsx` — two sheets: `Runs`, `Labels`

**How to run:**
```bash
Py/conv_env/Scripts/python Py/ConvertPkl.py Py/All_114_Timecourses.pkl Py/All_114_Arrays.pkl Py/All_114_Meta.xlsx
```

---

## Stage 4 — Functional Connectivity (`AssutaFC.py`)

**Runs on:** local (Anaconda3)

**Input:**
- `All_114_Arrays.pkl` — numpy arrays from Stage 3
- `All_114_Meta.xlsx` — Runs + Labels from Stage 3

**Output:** `All_114_FC.pkl` containing:

| Key      | Type                    | Description                                  |
|----------|-------------------------|----------------------------------------------|
| `FC`     | `(nRuns, 2, 114)` array | Fisher z-scored correlations                 |
| `Seeds`  | `['sum', 'normMean']`   | Axis-1 labels                                |
| `Runs`   | DataFrame               | Passed through from Meta.xlsx                |
| `Labels` | DataFrame               | Passed through from Meta.xlsx                |

**Seed (bilateral RSP = `SubParcel.startswith('Rsp')`):**
- `sum`: `L_timecourse + R_timecourse`
- `normMean`: `L_timecourse/|L| + R_timecourse/|R|` (equal hemisphere weight)

**Connectivity computation per run:**
1. Demean + unit-normalize each of the 114 parcel timecourses
2. Demean + unit-normalize each seed timecourse
3. Correlate: `seed (2, nTP) @ parcels.T (nTP, 114)` → `(2, 114)` Pearson r
4. Clip to `±(1 − 1e-4)`, apply `arctanh` → Fisher z

**How to run:**
```bash
C:\Anaconda3\python Py/AssutaFC.py Py/All_114_Arrays.pkl Py/All_114_Meta.xlsx Py/All_114_FC.pkl
```

---

## Stage 5 — Report Export (`AssutaExport.py`)

**Runs on:** local (Anaconda3)

**Input:**
- `All_114_FC.pkl` — output of Stage 4
- `hippocampi_estimates.csv` — demographics and hippocampal volumes

**`hippocampi_estimates.csv` columns:**
`SubjectID, Age, Sex, Diagnosis (CU/Preclinical/MCI/AD), lHipPredicted, lHipMeasured, rHipPredicted, rHipMeasured`

Predicted volumes are from a normative model (Potvin/FreeSurfer-based); measured are from the subject's FreeSurfer segmentation.

**Output:** `All_114_Report.xlsx` — two sheets:

**Sheet `Runs`** (~350 rows × ~239 cols):

| Column group        | Columns                                         |
|---------------------|-------------------------------------------------|
| Run descriptors     | `Subject, Run, TaskName, OrigID`                |
| Demographics        | `Age, Sex, Diagnosis, lHipPredicted, lHipMeasured, rHipPredicted, rHipMeasured` |
| FC (sum seed)       | `zCorr_RspSum_Reg1` … `zCorr_RspSum_Reg114`    |
| FC (normMean seed)  | `zCorr_RspAvg_Reg1` … `zCorr_RspAvg_Reg114`    |

Merge is `outer` on `Subject` ↔ `SubjectID`; subjects missing from either side get NaN.

**Sheet `Regions`** (114 rows): `Label, Hemi, Network, SubParcel, Desc`

Column `zCorr_*_Reg{Label}` in Runs maps to row with matching `Label` in Regions.

**Sanity check printed on run:**
- Subjects in runs only (no demographics)
- Subjects in demographics only (no runs)
- Subjects in both

**How to run:**
```bash
C:\Anaconda3\python Py/AssutaExport.py Py/All_114_FC.pkl MeitarData/hippocampi_estimates.csv Py/All_114_Report.xlsx
```

---

## Stage 6 — Interactive Visualization (`AssutaViz.py`)

**Runs on:** local (Anaconda3)

**Input:** `All_114_Report.xlsx` (from Stage 5)

**Plot:** Scatter — one point per subject (averaged across selected runs).

**Y-axis — Normalized Hippocampal Size** (radio buttons):

| Option      | Formula                                                     |
|-------------|-------------------------------------------------------------|
| Left        | `lHipMeasured / lHipPredicted`                              |
| Right       | `rHipMeasured / rHipPredicted`                              |
| Average     | `(lMeas/lPred + rMeas/rPred) / 2`                           |
| Cumulative  | `(lMeas + rMeas) / (lPred + rPred)`                         |

**X-axis — RSP Connectivity** (three controls):

| Control     | Type            | Default  | Description                                      |
|-------------|-----------------|----------|--------------------------------------------------|
| ClipZ       | text entry      | 3.0      | Clips z-values to `[−ClipZ, +ClipZ]` before averaging |
| Seed        | radio (Avg/Sum) | Avg      | Which seed variant to use                        |
| Tasks       | multi-select    | all−rest | Filter runs by task name before averaging        |
| Networks    | multi-select    | all 17   | Filter regions by Yeo network before averaging   |

**X aggregation order** (important — runs first, then regions):
1. For each region: average z across selected runs per subject
2. Average across selected regions → one X per subject

**Additional control:**
- **Diagnoses** (multi-select, all selected by default): filters which subjects are plotted

**Annotations:** Pearson r and p-value computed over all visible points, updated on every control change.

**Color coding:**

| Diagnosis   | Color  |
|-------------|--------|
| CU          | Green  |
| Preclinical | Blue   |
| MCI         | Violet |
| AD          | Red    |

**How to run:**
```bash
C:\Anaconda3\python Py/AssutaViz.py Py/All_114_Report.xlsx
```
