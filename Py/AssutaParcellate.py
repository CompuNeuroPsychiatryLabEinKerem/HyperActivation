# -*- coding: utf-8 -*-
import os, re, sys, pickle
op = os.path
opj = op.join

import numpy as np
import pandas as pd
import nibabel as nb
from glob import glob

YEO_DLABEL = r'C:\Projects\Parcellations\Yeo\Yeo2011_17Networks_91K.split_components.dlabel.nii'
N_PARCELS  = 114     # labels 1-114; label 0 is background


def loadParcelIdxs(dlabelFile=YEO_DLABEL):
    labelmap   = nb.load(dlabelFile).get_fdata().flatten().astype(int)
    parcelIdxs = [np.argwhere(labelmap == lbl).flatten() for lbl in np.arange(1, N_PARCELS+1)]
    return parcelIdxs


def parcellateFile(boldFile, parcelIdxs):
    boldData = nb.load(boldFile).get_fdata()    # (nTP, 91282)

    boldData -= boldData.mean(axis=0)           # remove per-vertex temporal mean
    boldData /= np.sqrt((boldData**2).mean())   # normalize by global RMS

    nTP      = boldData.shape[0]
    parcelTS = np.zeros((N_PARCELS, nTP))
    for i, idxs in enumerate(parcelIdxs):
        if idxs.size:
            parcelTS[i] = boldData[:, idxs].sum(axis=1)

    return parcelTS                             # (N_PARCELS, nTP)


def _parseRuns2Tasks(runs2tasksFile):
    # Returns list of (origID, trueSbj, fullRunIdx, taskName)
    # Line format: {trueSbj}-{fullRunIdx}-{taskName}__{boldFileSignature}_
    entries = []
    for line in open(runs2tasksFile):
        line = line.strip()
        if not line or '__' not in line:
            continue
        left, right  = line.split('__', 1)
        origID       = right.rstrip('_')        # 'sub-AvShB_ses-001_task-mental_run-003'
        idx1         = left.index('-')
        idx2         = left.index('-', idx1+1)
        trueSbj      = left[:idx1]
        fullRunIdx   = int(left[idx1+1:idx2])
        taskName     = left[idx2+1:]
        entries.append((origID, trueSbj, fullRunIdx, taskName))
    return entries


def parcellateDir(dtseriesDir, runs2tasksFile, outputFile, dlabelFile=YEO_DLABEL):
    parcelIdxs  = loadParcelIdxs(dlabelFile)
    parcelsSize = np.array([idxs.size for idxs in parcelIdxs])

    # All cleaned bold files keyed by signature (everything before _Atlas)
    allFiles    = glob(opj(dtseriesDir, '*_Atlas_s0_cleaned.dtseries.nii'))
    boldSigDict = {op.basename(f).split('_Atlas')[0]: f for f in allFiles}

    # --- Task runs from Runs2Tasks ---
    r2tEntries = _parseRuns2Tasks(runs2tasksFile)
    r2tSigSet  = {origID for origID, *_ in r2tEntries}

    taskRuns = []
    for origID, trueSbj, fullRunIdx, taskName in r2tEntries:
        if origID not in boldSigDict:
            print(f'  WARNING: no cleaned file found for {origID}')
            continue
        taskRuns.append((boldSigDict[origID], trueSbj, fullRunIdx, taskName, origID))

    # --- Rest runs (task-rest files not in Runs2Tasks) ---
    restRuns = []
    for sig, f in boldSigDict.items():
        if 'task-rest' not in sig or sig in r2tSigSet:
            continue
        sbjSess  = sig.split('_')[0].replace('sub-', '')   # e.g. 'AvShA'
        if sbjSess[-1] not in 'AB':
            print(f'  WARNING: unexpected session suffix in {sig} — skipping')
            continue
        trueSbj  = sbjSess[:-1]                             # e.g. 'AvSh'
        runIdx   = 1 if sbjSess.endswith('A') else 2
        restRuns.append((f, trueSbj, runIdx, 'rest', sig))

    # --- Process all ---
    allRuns = taskRuns + restRuns
    print(f'Processing {len(taskRuns)} task runs + {len(restRuns)} rest runs...', flush=True)

    timecourses = []
    rows        = []
    for boldFile, trueSbj, fullRunIdx, taskName, origID in allRuns:
        print(f'  {op.basename(boldFile)}', flush=True)
        timecourses.append(parcellateFile(boldFile, parcelIdxs))
        rows.append(dict(Subject=trueSbj, Run=fullRunIdx, TaskName=taskName, OrigID=origID))

    out = dict(
        Timecourses = timecourses,
        parcelsSize = parcelsSize,
        Runs        = pd.DataFrame(rows),
    )
    pickle.dump(out, open(outputFile, 'wb'))
    print(f'Saved {len(timecourses)} runs → {outputFile}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Parcellate cleaned dtseries files into Yeo parcel timeseries.')
    p.add_argument('dtseriesDir',    help='Folder containing *_Atlas_s0_cleaned.dtseries.nii files')
    p.add_argument('runs2tasksFile', help='Runs2Tasks.txt mapping file')
    p.add_argument('outputFile',     help='Output pickle file path')
    p.add_argument('--dlabel',       default=YEO_DLABEL, dest='dlabelFile',
                                     help='dlabel parcellation file (default: Yeo 91K)')
    args = p.parse_args()
    parcellateDir(args.dtseriesDir, args.runs2tasksFile, args.outputFile, args.dlabelFile)
