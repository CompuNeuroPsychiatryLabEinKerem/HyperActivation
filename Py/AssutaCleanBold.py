# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:50:26 2022

@author: uriur
Module: AssutaCleanRest
"""

import os, re, yaml, shutil, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
op = os.path
opj = op.join

import pandas as pd
import numpy as np
from glob import glob

import nibabel as nb
from sklearn.linear_model import LinearRegression as linreg
from scipy.interpolate import interp1d
from scipy.signal import iirfilter, sosfiltfilt

from CiftiUtils import setNumTpForCiftiHdr
from nilearn.image import resample_img
from collections import defaultdict as dd
from copy import deepcopy

r2z = lambda x: np.arctanh(np.clip(x, -(1-1e-6), 1-1e-6))

ALL_CONFOUNDS = ['motion_outlier*', 'aroma_motion*', 'cosine*', '[at]_comp_cor_*',
                 '(csf|global_signal|white_matter)(_derivative1)?(_power2)?',
                 '(rot|trans)_[xyz](_derivative1)?(_power2)?'
                 'csf_wm', '(std_)?dvars', 'framewise_displacement', 'tcompcor', 'rmsd']

USE_CONFOUNDS = ['global_signal', 'framewise_displacement', 'csf(_derivative)?',
                 'motion_outlier.+', '(rot|trans)_[xyz](_derivative1)?']

DEFAULT_OPTS = dict(
    MISC   = dict(TR=2.02, remove1stVols=2),
    SCRUB  = dict(Do=False, fdThr=-1, dvarsThr=75, minStreakThr=-1, acceptThr=-1),
    REGRESS= dict(Do=True,  useConfounds=USE_CONFOUNDS),
    FILTER = dict(Do=True,  iirOrder=3, band=[0.008, 0.25]),
)


def _validateFolders(bolddir, regdir, keyword, fwhm):
    fsuff    = f'_fwhm{fwhm}' if fwhm else ''
    boldGlob = glob(opj(bolddir, f'sub-*_ses-*_task-{keyword}_run-*_Atlas_s0{fsuff}.dtseries.nii'))
    cnfdGlob = glob(opj(regdir,  f'*_task-{keyword}_*_desc-confounds_*.tsv'))
    ok = True
    if not op.isdir(bolddir):
        print(f'ERROR: BOLD dir not found: {bolddir}')
        ok = False
    elif not boldGlob:
        print(f'ERROR: No matching dtseries.nii files in {bolddir}  (keyword={keyword}, fwhm={fwhm})')
        ok = False
    if not op.isdir(regdir):
        print(f'ERROR: Confounds dir not found: {regdir}')
        ok = False
    elif not cnfdGlob:
        print(f'ERROR: No matching confounds .tsv files in {regdir}  (keyword={keyword})')
        ok = False
    return ok


def cleanAllSubjects(cfgFile):
    cfg     = yaml.safe_load(open(cfgFile))
    paths   = cfg.pop('PATHS')
    root    = op.normpath(paths.get('root_dir', ''))
    _p      = lambda rel: op.normpath(opj(root, rel))   # joins root + relative; absolute paths pass through unchanged
    bolddir = _p(paths['bold_dir'])
    regdir  = _p(paths['confounds_dir'])
    dstdir  = _p(paths['dest_dir'])
    fwhm    = paths.get('fwhm', 0)
    keyword = paths.get('keyword', 'rest')  # "*" matches any keyword
    opts    = cfg                           # remaining keys: MISC, SCRUB, REGRESS, FILTER

    if not _validateFolders(bolddir, regdir, keyword, fwhm):
        return

    os.makedirs(dstdir, exist_ok=True)
    shutil.copy2(cfgFile, opj(dstdir, op.basename(cfgFile)))

    fsuff         = f'_fwhm{fwhm}' if fwhm else ''
    boldfiles     = glob(opj(bolddir, f'sub-*_ses-*_task-{keyword}_run-*_Atlas_s0{fsuff}.dtseries.nii'))
    confoundFiles = glob(opj(regdir,  f'*_task-{keyword}_*_desc-confounds_*.tsv'))

    # Key: everything up to the modality suffix — e.g. 'sub-HaHaA_ses-001_task-mental_run-001'
    boldfilesDict = {op.basename(f).split('_Atlas')[0]: f for f in boldfiles}
    cnfdfilesDict = {op.basename(f).split('_desc')[0]:  f for f in confoundFiles}

    boldSet    = set(boldfilesDict)
    cnfdSet    = set(cnfdfilesDict)
    onlyBold   = sorted(boldSet - cnfdSet)
    onlyConf   = sorted(cnfdSet - boldSet)
    commonKeys = sorted(boldSet & cnfdSet)

    print(f'\n--- Pre-run check ---')
    print(f'  BOLD files found:     {len(boldSet)}')
    print(f'  Confound files found: {len(cnfdSet)}')
    print(f'  Matched pairs:        {len(commonKeys)}')
    if onlyBold:
        print(f'  No confounds for:     {onlyBold}')
    if onlyConf:
        print(f'  No BOLD for:          {onlyConf}')
    if not commonKeys:
        print('  No matched pairs to process. Exiting.')
        return
    print(f'---------------------\n')

    sfx  = '.dtseries.nii'
    Lsfx = len(sfx)
    nDone = nSkipped = nFailed = 0

    # Separate already-done files (skip-if-exists) from work to dispatch
    toRun = []
    for key in commonKeys:
        boldFile      = boldfilesDict[key]
        confoundsFile = cnfdfilesDict[key]
        outFile       = boldFile[:-Lsfx] + '_cleaned' + sfx
        if op.exists(outFile):
            nSkipped += 1
        else:
            toRun.append((boldFile, confoundsFile, outFile))

    nWorkers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    nTotal   = len(toRun)
    print(f'Submitting {nTotal} subject/runs across {nWorkers} workers '
          f'({nSkipped} already exist)...', flush=True)

    with ProcessPoolExecutor(max_workers=nWorkers) as pool:
        futures = {pool.submit(cleanBoldFile, bold, cnfd, out, opts): op.basename(out)
                   for bold, cnfd, out in toRun}
        for future in as_completed(futures):
            name   = futures[future]
            n      = nDone + nFailed + 1
            try:
                status = future.result()
                if status == 'ok':
                    nDone += 1
                else:
                    nFailed += 1   # covers REJECTED and SKIPPED — both need attention
                print(f'  [{n}/{nTotal}] {status}: {name}', flush=True)
            except Exception as e:
                nFailed += 1
                print(f'  [{n}/{nTotal}] ERROR {name}: {e}', flush=True)

    print(f'\nDone: {nDone} cleaned, {nSkipped} skipped (already existed), {nFailed} failed.',
          flush=True)


def getOutliers(confoundsDf, remove1stVols=2, fdThr=-1, dvarsThr=-1,
                minStreakThr=-1, acceptThr=-1, **kwargs):

    numTrs    = len(confoundsDf)
    isOutlier = np.zeros(numTrs, dtype=bool)

    doScrubFD    = fdThr    > 0
    doScrubDVars = dvarsThr > 0
    numFdOutliers = numDvOutliers = 0
    if doScrubFD:
        isOutlier    |= (confoundsDf.framewise_displacement.values > fdThr)
        numFdOutliers = isOutlier.sum()
    if doScrubDVars:
        isOutlier     |= confoundsDf.dvars.values > dvarsThr
        numDvOutliers  = isOutlier.sum() - numFdOutliers
    if minStreakThr > 0:
        isOutlier = markOutliersSeries(isOutlier, thr=minStreakThr)

    if 0 < acceptThr < 1:
        areEnoughInliers = (1 - isOutlier.sum()/isOutlier.size) > acceptThr
    else:
        areEnoughInliers = True

    iInliers = np.argwhere(1-isOutlier[remove1stVols:]).flatten() + remove1stVols
    if iInliers.size:
        firstInlier = iInliers[0]
        lastInlier  = iInliers[-1]
        nRemoveHead = firstInlier
        nRemoveTail = numTrs - (lastInlier+1)
        isOutlier   = isOutlier[nRemoveHead:numTrs-nRemoveTail]

        iOutliers = np.argwhere( isOutlier).flatten()
        iInliers  = np.argwhere(~isOutlier).flatten()
    else:
        nRemoveHead, nRemoveTail = 600, 600
        iOutliers = iInliers = np.array([])

    return iOutliers, iInliers, nRemoveHead, nRemoveTail, areEnoughInliers


def markOutliersSeries(isOutlier, thr):
    if not isOutlier.sum():
        return isOutlier
    padded       = np.hstack((1, isOutlier, 1))
    iOutliers    = np.argwhere(padded).flatten()
    iShortSeries = np.argwhere(np.diff(iOutliers) <= thr).flatten()
    if not iShortSeries.size:
        return isOutlier

    iNewVolsToRemove = np.hstack(tuple(np.arange(iOutliers[i]+1, iOutliers[i+1]) for i in iShortSeries)) - 1
    resOutliers = isOutlier.copy()
    resOutliers[iNewVolsToRemove] = 1

    return resOutliers


def getOpts(optsInp={}):
    opts = deepcopy(DEFAULT_OPTS)
    for key, subopts in optsInp.items():
        if key in opts:
            opts[key].update(subopts)
    return opts


def _cleanBoldData(boldFile, confoundsFile, opts):
    """Clean BOLD in memory. Returns (boldData, status); boldData is None on failure."""
    remove1stVols = opts['MISC']['remove1stVols']
    TR            = opts['MISC']['TR']
    SCRUB         = opts['SCRUB']
    REGRESS       = opts['REGRESS']
    FILTER        = opts['FILTER']

    obj      = nb.load(boldFile, mmap=False)
    boldData = obj.get_fdata()
    numTrs   = boldData.shape[0]

    confoundsDf = pd.read_csv(confoundsFile, sep='\t')

    if len(confoundsDf) != numTrs:
        return None, obj, f'SKIPPED — TR mismatch: BOLD={numTrs}, confounds={len(confoundsDf)}'

    doScrub = SCRUB['Do']
    if doScrub:
        outlierData = getOutliers(confoundsDf, remove1stVols=remove1stVols, **SCRUB)
        iOutliers, iInliers, nRemoveHead, nRemoveTail, areEnoughInliers = outlierData
    else:
        nRemoveHead, nRemoveTail, areEnoughInliers = remove1stVols, 0, True

    if not areEnoughInliers:
        numOutliers = iOutliers.size + nRemoveHead + nRemoveTail
        return None, obj, f'REJECTED — too many outliers ({numOutliers} TRs)'

    if doScrub and not iOutliers.size:
        doScrub = False

    boldData = boldData[nRemoveHead:numTrs-nRemoveTail, :]
    if doScrub:
        boldData = boldData[iInliers, :]

    if REGRESS['Do']:
        useRegressors = REGRESS['useConfounds']
        regressors    = getRegressors(confoundsDf, useRegressors).values
        regressors    = regressors[nRemoveHead:numTrs-nRemoveTail, :]
        if doScrub:
            regressors = regressors[iInliers, :]
        regobj    = linreg(fit_intercept=True).fit(regressors, boldData)
        boldData -= regobj.predict(regressors)

    if FILTER['Do']:
        numTrsNoHeadTail = numTrs - nRemoveHead - nRemoveTail
        if doScrub:
            fi       = interp1d(iInliers, boldData, kind='cubic', axis=0)
            boldData = fi(range(numTrsNoHeadTail))

        order, band = FILTER['iirOrder'], FILTER['band']
        wn, btype   = (band, 'bandpass') if band[-1] < 1/(2*TR) else (band[0], 'highpass')
        sos      = iirfilter(order, analog=False, fs=1/TR, output='sos', Wn=wn, btype=btype)
        boldData = sosfiltfilt(sos, boldData, axis=0)

        if doScrub:
            boldData = boldData[iInliers, :]

    return boldData, obj, 'ok'


def cleanBoldFile(boldFile, confoundsFile, outFile, opts={}):
    opts             = getOpts(opts)
    boldData, obj, status = _cleanBoldData(boldFile, confoundsFile, opts)
    if status != 'ok':
        return status

    chdr = setNumTpForCiftiHdr(obj.header, boldData.shape[0])
    nb.Cifti2Image(dataobj=boldData, header=chdr,
                   nifti_header=obj.nifti_header, extra=obj.extra,
                   file_map=obj.file_map).to_filename(outFile)
    return 'ok'


def getRegressors(confoundsDf, useRegressors):
    allCols    = set(confoundsDf.columns)
    chosenCols = []

    isReg = lambda txt: not re.match(r'\A\w+\Z', txt)

    for cfdName in useRegressors:
        if isReg(cfdName):
            reg = re.compile(cfdName)
            chosenCols += [col for col in allCols if reg.match(col)]
        elif cfdName in allCols:
            chosenCols.append(cfdName)
        else:
            assert f"Couldn't find regressor {cfdName}! Exitting..."

    df = confoundsDf[chosenCols]
    return df


def checkRestFile(img2flip='yeo'):
    resftFile   = opj(op.sep, 'temp', 'sub-AlZi_ses-01_task-rest_run-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz')
    yeofile     = opj('C:', op.sep, 'Projects', 'Parcellations', 'Yeo', 'Yeo2011_17Networks_N1000.split_components.FSL_MNI152_2mm.nii.gz')
    yeotextfile = opj(op.sep, 'Projects', 'Parcellations', 'Yeo', '17Networks_ColorLUT_freeview.txt')

    parcels = dd(dict)
    for line in open(yeotextfile):
        num, desc = line.split()[:2]
        num = int(num)
        if num == 0:
            continue
        parts = desc.split('_')
        hm    = parts[1][0]
        name  = '_'.join(parts[2:])
        parcels[name][hm] = num

    restobj  = nb.load(resftFile)
    restData = restobj.get_fdata().mean(-1)

    yeoobj  = nb.load(yeofile)
    yeoData = yeoobj.get_fdata()

    if img2flip == 'yeo':
        yeoData = resample_img(yeoobj, restobj.affine, restobj.shape[:-1], interpolation='nearest').get_fdata()
    else:
        restData = resample_img(nb.Nifti1Image(restData, restobj.affine), yeoobj.affine, yeoobj.shape).get_fdata()
    yeoData  = yeoData.flatten()
    restData = restData.flatten()

    numPrcls    = len(parcels)
    parcelsMag  = dict(L=np.zeros(numPrcls), R=np.zeros(numPrcls))
    parcelsStd  = dict(L=np.zeros(numPrcls), R=np.zeros(numPrcls))
    parcelNames = []
    for cnt, (name, idxs) in enumerate(parcels.items()):
        parcelNames.append(name)
        for hm, num in idxs.items():
            pinds               = np.argwhere(yeoData==num).flatten()
            parcelsMag[hm][cnt] = restData[pinds].mean()
            parcelsStd[hm][cnt] = restData[pinds].std()

    return parcelsMag, parcelsStd, parcelNames


def getNumWeakVertices(bolddir, fwhm=0, vertexThr=0.5):
    yeoDataFile  = opj('C:', op.sep, 'Projects', 'Parcellations', 'Yeo', 'Yeo2011_17Networks.split_components.dscalar.nii')
    yeoNamesFile = opj('C:', op.sep, 'Projects', 'Parcellations', 'Yeo', '17Networks_ColorLUT_freeview.txt')
    yeoCortexData = readMeanCortex(yeoDataFile)
    nzIdxs        = np.argwhere(yeoCortexData>0).flatten()
    yeoCortexData = yeoCortexData[nzIdxs]

    readopts = dict(index_col=None, skiprows=1, delim_whitespace=True, header=None)
    yeoAreaIdx2Names = dict(pd.read_csv(yeoNamesFile, **readopts)[[0,1]].values)
    yeoAreaIdx2Names = {k: '_'.join(v.split('_')[1:]) for k, v in yeoAreaIdx2Names.items()}

    yeoAreaIdxs = dd(list)
    for i, v in enumerate(yeoCortexData):
        yeoAreaIdxs[v].append(i)
    yeoAreaIdxs = {yeoAreaIdx2Names[k]: np.array(v) for k, v in yeoAreaIdxs.items()}

    fsuff     = f'_fwhm{fwhm}' if fwhm else ''
    restfiles = glob(opj(bolddir, f'*-rest-1-cifti_sbj_Atlas_s0{fsuff}.dtseries.nii'))
    subjects        = []
    numWeakVertices = []
    corruptedAreas  = []

    for restfile in restfiles:
        sbj = op.basename(restfile).split('-')[0]
        subjects.append(sbj)
        sbjdata = readMeanCortex(restfile)[nzIdxs]
        thr     = getWeakThr(sbjdata)

        binSbjData = sbjdata < thr
        numWeakVertices.append(binSbjData.sum())
        corruptedAreas.append(getCorruptedAreas(binSbjData, yeoAreaIdxs, thr=vertexThr))

    numWeakVertices = np.array(numWeakVertices)
    return numWeakVertices, corruptedAreas, subjects


def getCorruptedAreas(binSbjData, yeoAreaIdxs, thr=0.5):
    percentMissing = {}
    for name, idxs in yeoAreaIdxs.items():
        percentMissing[name] = binSbjData[idxs].sum() / idxs.size

    areasCorrupted = set(name for name, p in percentMissing.items() if p > thr)
    return areasCorrupted


def readMeanCortex(file):
    obj = nb.load(file)
    bms = list(obj.header.matrix[1].brain_models)
    if obj.shape[0] > 1:
        data = obj.get_fdata().mean(0).flatten()
    else:
        data = obj.get_fdata().flatten()

    hemisData = []
    for bm in bms[:2]:
        hmdata = np.zeros(bm.surface_number_of_vertices)
        i0     = bm.index_offset
        i1     = i0 + bm.index_count
        hmdata[bm.vertex_indices] = data[i0:i1]
        hemisData.append(hmdata)

    hemisData = np.hstack(hemisData)
    return hemisData


def getWeakThr(data, athr=0.1):
    refVal = np.percentile(data, 75)
    thr    = refVal * athr
    return thr


"""
if True:
   fig, (ax1, ax2) = plt.subplots(2,1)
   w = 0.4
   x = np.arange(len(netConnectionalHm))
   for ax, data, thr in zip([ax1, ax2], [pairedConnectionalHm, netConnectionalHm], [0.25, 0.5]):
       for ihm, hdata in enumerate(zip(*data)):
           offset = w*(ihm-0.5)
           ax.bar(x-offset, hdata, width=w)
           ax.plot(x, thr*np.ones(36), 'r')

if True:
    fig, (ax1, ax2) = plt.subplots(1,2)
    w = 0.25
    for ax, data in zip([ax1, ax2], [pairedCon, netCon]):
        ax.set_xticks([0,1])
        ax.set_xticklabels(['L', 'R'])
        for i, (dat, label) in enumerate(zip(data, ['FSL', 'FREP', 'FPREP_12'])):
            ax.bar(np.arange(0,2) + w*(i-1), dat[0], width=w, label=label)
    ax1.set_title('Paired Connectional Homogeniety')
    ax2.set_title('Net Connectional Homogeniety')
    ax1.legend()
"""

if __name__ == '__main__':
    cfgFile = sys.argv[1] if len(sys.argv) > 1 else 'CleanCfg.yaml'
    cleanAllSubjects(cfgFile)
