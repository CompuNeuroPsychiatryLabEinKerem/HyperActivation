# -*- coding: utf-8 -*-
import os, re, sys, pickle
op = os.path
opj = op.join

import numpy as np
import pandas as pd
import nibabel as nb
from glob import glob

YEO_DLABEL = r'C:\Projects\Parcellations\Yeo\Yeo2011_17Networks_91K.split_components.dlabel.nii'

def _parseDesc(desc):
    # '17Networks_LH_SomMotA'       -> Hemi='L', Network='SomMotA', SubParcel=''
    # '17Networks_LH_SomMotB_Cent'  -> Hemi='L', Network='SomMotB', SubParcel='Cent'
    parts    = desc.replace('17Networks_', '').split('_')
    hemi     = parts[0][0]                          # 'L' or 'R'
    network  = parts[1]
    subParcel = parts[2] if len(parts) > 2 else ''
    return hemi, network, subParcel


def loadParcelIdxs(dlabelFile, labelFile):
    X            = nb.load(dlabelFile, mmap=False).get_fdata().flatten().astype(int)
    parcelLabels = np.unique(X)[1:]                 # drop background (0)
    parcelIdxs   = [np.argwhere(X == lbl).flatten() for lbl in parcelLabels]

    label2Desc = {}
    for line in open(labelFile):
        parts = line.split()
        if len(parts) >= 2 and parts[1] != 'NONE':
            label2Desc[int(parts[0])] = parts[1]

    rows = []
    for lbl in parcelLabels:
        desc                   = label2Desc.get(int(lbl), 'Unknown')
        hemi, network, subParcel = _parseDesc(desc)
        rows.append(dict(Label=int(lbl), Hemi=hemi, Network=network,
                         SubParcel=subParcel, Desc=desc))

    labelsDF = pd.DataFrame(rows)
    return parcelIdxs, labelsDF


def parcellateFile(boldFile, parcelIdxs):
    try:
        boldData = nb.load(boldFile, mmap=False).get_fdata()    # (nTP, 91282)
    except OSError as e:
        print(f'  ERROR loading {op.basename(boldFile)}: {e}', flush=True)
        return None

    boldData -= boldData.mean(axis=0)           # remove per-vertex temporal mean
    boldData /= np.sqrt((boldData**2).mean())   # normalize by global RMS

    nTP      = boldData.shape[0]
    nParcels = len(parcelIdxs)
    parcelTS = np.zeros((nParcels, nTP))
    for i, idxs in enumerate(parcelIdxs):
        if idxs.size:
            parcelTS[i] = boldData[:, idxs].sum(axis=1)

    return parcelTS                             # (nParcels, nTP)


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


def _sigToMeta(sig):
    # sig: 'sub-AvSh_ses-001_task-mental_run-001' → (subject, taskName, runIdx)
    parts    = {k: v for k, v in (p.split('-', 1) for p in sig.split('_') if '-' in p)}
    return parts.get('sub', sig), parts.get('task', ''), int(parts.get('run', 0))


def parcellateDir(dtseriesDir, labelFile, outputPref, runs2tasksFile=None, dlabelFile=YEO_DLABEL):
    parcelIdxs, labelsDF = loadParcelIdxs(dlabelFile, labelFile)
    parcelsSize          = np.array([idxs.size for idxs in parcelIdxs])

    # All cleaned bold files keyed by signature (everything before _Atlas)
    allFiles    = glob(opj(dtseriesDir, '*_Atlas_s0_*.dtseries.nii'))
    boldSigDict = {op.basename(f).split('_Atlas')[0]: f for f in allFiles}

    if runs2tasksFile is None:
        # No mapping file: parse Subject/Task/Run directly from filename
        allRuns = []
        for sig, f in sorted(boldSigDict.items()):
            trueSbj, taskName, runIdx = _sigToMeta(sig)
            allRuns.append((f, trueSbj, runIdx, taskName, sig))
        print(f'Processing {len(allRuns)} runs (no Runs2Tasks mapping) '
              f'({len(parcelIdxs)} parcels)...', flush=True)
    else:
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
            sbjSess = sig.split('_')[0].replace('sub-', '')     # e.g. 'AvShA'
            if len(sbjSess) == 4:
                print(f'  INFO: single-session subject {sbjSess}, assigning Run=1')
                trueSbj, runIdx = sbjSess, 1
            elif sbjSess[-1] not in 'AB':
                print(f'  WARNING: unexpected session suffix in {sig} — skipping')
                continue
            else:
                trueSbj = sbjSess[:-1]                          # e.g. 'AvSh'
                runIdx  = 1 if sbjSess.endswith('A') else 2
            restRuns.append((f, trueSbj, runIdx, 'rest', sig))

        allRuns = taskRuns + restRuns
        print(f'Processing {len(taskRuns)} task runs + {len(restRuns)} rest runs '
              f'({len(parcelIdxs)} parcels)...', flush=True)

    timecourses = []
    rows        = []
    for boldFile, trueSbj, fullRunIdx, taskName, origID in allRuns:
        print(f'  {op.basename(boldFile)}', flush=True)
        ts = parcellateFile(boldFile, parcelIdxs)
        if ts is None:
            continue
        timecourses.append(ts)
        rows.append(dict(Subject=trueSbj, Run=fullRunIdx, TaskName=taskName, OrigID=origID))

    arraysFile = f'{outputPref}_Arrays.pkl'
    metaFile   = f'{outputPref}_Meta.xlsx'

    pickle.dump({'Timecourses': timecourses, 'parcelsSize': parcelsSize},
                open(arraysFile, 'wb'))

    with pd.ExcelWriter(metaFile, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='Runs',   index=False)
        labelsDF.to_excel(           writer, sheet_name='Labels', index=False)

    print(f'Saved {len(timecourses)} runs → {arraysFile}, {metaFile}', flush=True)


def runAssutaGreg():
    # dtseriesDir, labelFile, outputFile, runs2tasksFile=None, dlabelFile=YEO_DLABEL
    parcellateDir(
        dtseriesDir = r'C:\Projects\Assuta\NO_BBR\Task_Half_Cleaned',
        labelFile = r'\Projects\Parcellations\Yeo\17Networks_ColorLUT_freeview.txt',
        outputPref = 'Greg_114HC',
        )


if __name__ == '__main__':
    doRunWithCMD = False
    doRunGreg = True
    if doRunWithCMD:
        import argparse
        p = argparse.ArgumentParser(description='Parcellate cleaned dtseries files into Yeo parcel timeseries.')
        p.add_argument('dtseriesDir',  help='Folder containing *_Atlas_s0_cleaned.dtseries.nii files')
        p.add_argument('labelFile',    help='Parcel label text file (parcelIdx, desc, r, g, b, a)')
        p.add_argument('outputPref',   help='Output file prefix (produces <pref>_Arrays.pkl and <pref>_Meta.xlsx)')
        p.add_argument('--runs2tasks', default=None, dest='runs2tasksFile',
                                       help='Runs2Tasks.txt mapping file (omit for single-session datasets)')
        p.add_argument('--dlabel',     default=YEO_DLABEL, dest='dlabelFile',
                                       help='dlabel parcellation file (default: Yeo 91K)')
        args = p.parse_args()
        parcellateDir(args.dtseriesDir, args.labelFile, args.outputPref,
                      args.runs2tasksFile, args.dlabelFile)
        
    if doRunGreg:
        runAssutaGreg()
