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


def computeFC(pklFile, outputFile=None):
    data        = pickle.load(open(pklFile, 'rb'))
    timecourses = data['Timecourses']   # list of (nParcels, nTP)
    parcelsSize = data['parcelsSize']   # (nParcels,)
    labelsDF    = data['Labels']
    runsDF      = data['Runs']

    rspDF = labelsDF[labelsDF.Network == 'Rsp']
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
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Compute Rsp seed-based FC from parcel timecourses.')
    p.add_argument('pklFile',    help='Input parcellation pickle (from AssutaParcellate.py)')
    p.add_argument('outputFile', help='Output FC pickle file path')
    args = p.parse_args()
    computeFC(args.pklFile, args.outputFile)
