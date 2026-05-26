# -*- coding: utf-8 -*-
import os, sys, pickle, yaml, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
op  = os.path
opj = op.join

import numpy as np
import pandas as pd
from glob import glob

from AssutaCleanBold  import _cleanBoldData, getOpts
from AssutaParcellate import loadParcelIdxs, _sigToMeta, _parseRuns2Tasks, YEO_DLABEL

SAVE_EVERY = 5


# ── Worker (runs in subprocess) ────────────────────────────────────────────────

def _processFile(boldFile, confoundsFile, opts, parcelIdxs):
    boldData, _obj, status = _cleanBoldData(boldFile, confoundsFile, opts)
    if boldData is None:
        return None, status

    boldData -= boldData.mean(axis=0)
    boldData /= np.sqrt((boldData**2).mean())

    nTP, nParcels = boldData.shape[0], len(parcelIdxs)
    parcelTS = np.zeros((nParcels, nTP))
    for i, idxs in enumerate(parcelIdxs):
        if idxs.size:
            parcelTS[i] = boldData[:, idxs].sum(axis=1)
    return parcelTS, 'ok'


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _checkpoint(outputPref, timecourses, rows, parcelsSize, labelsDF):
    pickle.dump({'Timecourses': timecourses, 'parcelsSize': parcelsSize},
                open(f'{outputPref}_Arrays.pkl', 'wb'))
    with pd.ExcelWriter(f'{outputPref}_Meta.xlsx', engine='openpyxl') as w:
        pd.DataFrame(rows).to_excel(w, sheet_name='Runs',   index=False)
        labelsDF.to_excel(           w, sheet_name='Labels', index=False)


def _buildRunList(boldDir, confoundsDir, keyword, fwhm, runs2tasksFile, log):
    fsuff     = f'_fwhm{fwhm}' if fwhm else ''
    boldFiles = glob(opj(boldDir,      f'sub-*_ses-*_task-{keyword}_run-*_Atlas_s0{fsuff}.dtseries.nii'))
    cnfdFiles = glob(opj(confoundsDir, f'*_task-{keyword}_*_desc-confounds_*.tsv'))

    boldDict = {op.basename(f).split('_Atlas')[0]: f for f in boldFiles}
    cnfdDict = {op.basename(f).split('_desc')[0]:  f for f in cnfdFiles}

    boldSet, cnfdSet = set(boldDict), set(cnfdDict)
    log.append(f'MATCH  {len(boldSet)} bold  |  {len(cnfdSet)} confound files')
    for s in sorted(boldSet - cnfdSet): log.append(f'WARN   No confounds: {s}')
    for s in sorted(cnfdSet - boldSet): log.append(f'WARN   No BOLD:      {s}')

    common = sorted(boldSet & cnfdSet)
    log.append(f'MATCH  {len(common)} paired runs')

    if runs2tasksFile:
        r2tDict   = {origID: (sbj, idx, task)
                     for origID, sbj, idx, task in _parseRuns2Tasks(runs2tasksFile)}
        allRuns = []
        for sig in common:
            if sig in r2tDict:
                trueSbj, runIdx, taskName = r2tDict[sig]
            elif 'task-rest' in sig:
                sbjSess = sig.split('_')[0].replace('sub-', '')
                if sbjSess[-1] not in 'AB':
                    log.append(f'WARN   Unexpected session suffix, skipping: {sig}')
                    continue
                trueSbj  = sbjSess[:-1]
                runIdx   = 1 if sbjSess.endswith('A') else 2
                taskName = 'rest'
            else:
                log.append(f'WARN   No R2T entry for non-rest run, skipping: {sig}')
                continue
            allRuns.append((boldDict[sig], cnfdDict[sig], sig, trueSbj, runIdx, taskName, sig))
    else:
        allRuns = []
        for sig in common:
            trueSbj, taskName, runIdx = _sigToMeta(sig)
            allRuns.append((boldDict[sig], cnfdDict[sig], sig, trueSbj, runIdx, taskName, sig))

    return allRuns


# ── Pipeline ───────────────────────────────────────────────────────────────────

def runPipeline(cfgFile, outputPref):
    cfg      = yaml.safe_load(open(cfgFile))
    paths    = cfg.pop('PATHS')
    parcCfg  = cfg.pop('PARCELLATE', {})
    root     = op.normpath(paths.get('root_dir', ''))
    _p       = lambda rel: op.normpath(opj(root, rel))

    boldDir      = _p(paths['bold_dir'])
    confoundsDir = _p(paths['confounds_dir'])
    keyword      = paths.get('keyword', '*')
    fwhm         = paths.get('fwhm', 0)
    dlabelFile   = parcCfg.get('dlabelFile', YEO_DLABEL)
    labelFile    = parcCfg['labelFile']
    r2tFile      = parcCfg.get('runs2tasksFile', None)

    opts = getOpts(cfg)   # merge with defaults once; pass to workers

    log = []
    parcelIdxs, labelsDF = loadParcelIdxs(dlabelFile, labelFile)
    parcelsSize = np.array([idxs.size for idxs in parcelIdxs])

    allRuns = _buildRunList(boldDir, confoundsDir, keyword, fwhm, r2tFile, log)
    nWorkers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    print(f'Processing {len(allRuns)} runs across {nWorkers} workers '
          f'({len(parcelIdxs)} parcels)...', flush=True)

    timecourses, rows = [], []
    failedRuns        = []
    pendingSigs       = {sig for _, _, sig, *_ in allRuns}
    nSinceChk         = 0

    # ── Parallel phase ─────────────────────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=nWorkers) as pool:
        futures = {
            pool.submit(_processFile, bold, cnfd, opts, parcelIdxs):
                (bold, cnfd, sig, trueSbj, runIdx, taskName, origID)
            for bold, cnfd, sig, trueSbj, runIdx, taskName, origID in allRuns
        }

        for future in as_completed(futures):
            bold, cnfd, sig, trueSbj, runIdx, taskName, origID = futures[future]
            pendingSigs.discard(sig)
            name = op.basename(bold)

            try:
                parcelTS, status = future.result()
            except Exception as e:
                parcelTS, status = None, f'EXCEPTION: {e}'

            if parcelTS is not None:
                timecourses.append(parcelTS)
                rows.append(dict(Subject=trueSbj, Run=runIdx, TaskName=taskName, OrigID=origID))
                nSinceChk += 1
                if nSinceChk >= SAVE_EVERY:
                    _checkpoint(outputPref, timecourses, rows, parcelsSize, labelsDF)
                    nSinceChk = 0
                print(f'  OK   {name}', flush=True)
            else:
                log.append(f'FAIL   [{sig}] {status}')
                failedRuns.append((bold, cnfd, sig, trueSbj, runIdx, taskName, origID))
                print(f'  FAIL {name}: {status}', flush=True)

    # Files that never yielded a future (pool died mid-run)
    allRunsLookup = {sig: (bold, cnfd, trueSbj, runIdx, taskName, origID)
                     for bold, cnfd, sig, trueSbj, runIdx, taskName, origID in allRuns}

    for sig in pendingSigs:
        log.append(f'LOST   [{sig}] never completed in parallel phase')
        bold, cnfd, trueSbj, runIdx, taskName, origID = allRunsLookup[sig]
        failedRuns.append((bold, cnfd, sig, trueSbj, runIdx, taskName, origID))

    # ── Serial retry ───────────────────────────────────────────────────────────
    if failedRuns:
        print(f'\nSerial retry: {len(failedRuns)} files...', flush=True)
        for bold, cnfd, sig, trueSbj, runIdx, taskName, origID in failedRuns:
            print(f'  Retry {op.basename(bold)}', flush=True)
            try:
                parcelTS, status = _processFile(bold, cnfd, opts, parcelIdxs)
            except Exception as e:
                parcelTS, status = None, f'EXCEPTION: {e}'

            if parcelTS is not None:
                timecourses.append(parcelTS)
                rows.append(dict(Subject=trueSbj, Run=runIdx, TaskName=taskName, OrigID=origID))
                log.append(f'RETRY-OK   [{sig}]')
                print(f'    → OK', flush=True)
            else:
                log.append(f'RETRY-FAIL [{sig}] {status}')
                print(f'    → FAIL: {status}', flush=True)

    # ── Final save ─────────────────────────────────────────────────────────────
    _checkpoint(outputPref, timecourses, rows, parcelsSize, labelsDF)
    shutil.copy2(cfgFile, f'{outputPref}_Config.yaml')

    nOK   = len(timecourses)
    nFail = sum(1 for l in log if l.startswith('RETRY-FAIL') or
                (l.startswith('FAIL') and 'RETRY' not in l))
    log.append(f'SUMMARY  {nOK} completed, {nFail} failed')

    with open(f'{outputPref}_Log.txt', 'w') as f:
        f.write('\n'.join(log) + '\n')

    print(f'\nDone: {nOK} runs → {outputPref}_Arrays.pkl / _Meta.xlsx', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python AssutaCleanParc.py <cfgFile> <outputPref>')
        sys.exit(1)
    runPipeline(sys.argv[1], sys.argv[2])
