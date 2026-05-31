# -*- coding: utf-8 -*-
import sys, pickle
import numpy as np
import pandas as pd
import nibabel as nb

from AssutaParcellate import loadParcelIdxs, YEO_DLABEL

LABEL_FILE = r'C:\Projects\Parcellations\Yeo\17Networks_ColorLUT_freeview.txt'


def buildStdMap(arraysPkl, metaXlsx, outputFile,
                dlabelFile=YEO_DLABEL, labelFile=LABEL_FILE):

    arrays      = pickle.load(open(arraysPkl, 'rb'))
    timecourses = arrays['Timecourses']          # list of (nParcels, nTP)
    runsDF      = pd.read_excel(metaXlsx, sheet_name='Runs')

    parcelIdxs, _ = loadParcelIdxs(dlabelFile, labelFile)
    nVertices      = nb.load(dlabelFile, mmap=False).get_fdata().shape[1]
    nParcels       = len(parcelIdxs)

    # ── RMS per run → average per subject ─────────────────────────────────────
    subjects = runsDF['Subject'].tolist()
    rmsRuns  = np.stack([np.sqrt((ts**2).mean(axis=1)) for ts in timecourses])
    # rmsRuns: (nRuns, nParcels)

    sbjOrder  = list(dict.fromkeys(subjects))   # unique subjects, preserving order
    sbjRms    = np.zeros((len(sbjOrder), nParcels))
    for si, sbj in enumerate(sbjOrder):
        mask = [s == sbj for s in subjects]
        sbjRms[si] = rmsRuns[mask].mean(axis=0)

    # ── 0-order interpolation → 91K vertices ──────────────────────────────────
    sbjMaps = np.zeros((len(sbjOrder), nVertices))
    for pi, idxs in enumerate(parcelIdxs):
        if idxs.size:
            sbjMaps[:, idxs] = sbjRms[:, pi, np.newaxis]

    # ── Build dscalar CIFTI ────────────────────────────────────────────────────
    dlabel_img  = nb.load(dlabelFile, mmap=False)
    brain_axis  = dlabel_img.header.get_axis(1)
    scalar_axis = nb.cifti2.ScalarAxis(sbjOrder)
    header      = nb.Cifti2Header.from_axes((scalar_axis, brain_axis))

    nb.Cifti2Image(sbjMaps.astype(np.float32), header=header).to_filename(outputFile)
    print(f'Saved {len(sbjOrder)} subject maps → {outputFile}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4, 5):
        print('Usage: python AssutaStdMap.py <arraysPkl> <metaXlsx> <outputDscalar> '
              '[dlabelFile]')
        sys.exit(1)
    buildStdMap(*sys.argv[1:])
