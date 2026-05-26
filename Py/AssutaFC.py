# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd

CLIP = 1 - 1e-4


def _normTS(ts):
    # ts: (n, nTP) — demean and unit-norm along time axis
    ts   = ts - ts.mean(axis=1, keepdims=True)
    nrms = np.sqrt((ts**2).sum(axis=1, keepdims=True))
    nrms[nrms == 0] = 1
    return ts / nrms


def computeFC(arraysPkl=None, metaXlsx=None, outputFile=None, 
              filePref=None, returnResults=False):
    
    if filePref is not None:
        if arraysPkl is None:
            arraysPkl = f'{filePref}_Arrays.pkl'
        if metaXlsx is None:
            metaXlsx = f'{filePref}_Meta.xlsx'
        if outputFile is None:
            outputFile = f'{filePref}_FC.pkl'
            
    if arraysPkl is None:
        returnResults = True
            
    arrays      = pickle.load(open(arraysPkl, 'rb'))
    timecourses = arrays['Timecourses']   # list of (nParcels, nTP)
    parcelsSize = arrays['parcelsSize']   # (nParcels,)
    labelsDF    = pd.read_excel(metaXlsx, sheet_name='Labels')
    runsDF      = pd.read_excel(metaXlsx, sheet_name='Runs')

    rspDF = labelsDF[labelsDF.SubParcel.str.startswith('Rsp', na=False)]
    assert len(rspDF) == 2, f'Expected 2 Rsp parcels (L+R), found {len(rspDF)}'
    lIdx  = rspDF[rspDF.Hemi == 'L'].index[0]
    rIdx  = rspDF[rspDF.Hemi == 'R'].index[0]
    lSz   = parcelsSize[lIdx]
    rSz   = parcelsSize[rIdx]

    nFiles   = len(timecourses)
    nParcels = len(labelsDF)
    fc       = np.zeros((nFiles, 2, nParcels))

    for i, ts in enumerate(timecourses):
        ts_n = _normTS(ts)                              # (nParcels, nTP)

        L = ts[lIdx]; R = ts[rIdx]
        seeds   = np.stack([L + R,
                            L / lSz + R / rSz])         # (2, nTP)
        seeds_n = _normTS(seeds)

        r      = seeds_n @ ts_n.T                       # (2, nParcels)
        fc[i]  = np.arctanh(np.clip(r, -CLIP, CLIP))

    out = dict(FC=fc, Seeds=['sum', 'normMean'], Runs=runsDF, Labels=labelsDF)
    if outputFile:
        pickle.dump(out, open(outputFile, 'wb'))
        print(f'Saved FC ({nFiles} runs, 2 seeds, {nParcels} parcels) → {outputFile}', flush=True)

    if returnResults:
        return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Compute Rsp seed-based FC from parcel timecourses.')
    p.add_argument('filePref',     nargs='?', default=None,
                   help='File prefix — derives _Arrays.pkl, _Meta.xlsx, _FC.pkl automatically')
    p.add_argument('--arraysPkl',  default=None, help='Override arrays pkl path')
    p.add_argument('--metaXlsx',   default=None, help='Override meta xlsx path')
    p.add_argument('--outputFile', default=None, help='Override output FC pkl path')
    args = p.parse_args()
    computeFC(arraysPkl=args.arraysPkl, metaXlsx=args.metaXlsx,
              outputFile=args.outputFile, filePref=args.filePref)
