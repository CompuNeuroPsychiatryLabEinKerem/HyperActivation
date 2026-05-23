# -*- coding: utf-8 -*-
# Module: AssutaGLM
"""
Created on Tue Dec 21 13:25:32 2021

@author: uriur
"""

import os, re
op = os.path
opj = op.join

import pandas as pd
import numpy as np
from glob import glob
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
import nibabel as nb
from toolz import get
import subprocess
import json
from collections import defaultdict as dd
from random import shuffle



BASE_DIR = r'\Projects\Assuta\NO_BBR'
BOLD_DIR = opj(BASE_DIR, 'Final')
CONTRASTS_DIR = opj(BASE_DIR, 'Contrasts')
PARSED_LOGS_FILE_NAME = 'AllParsedLogs.csv'
COLUMN_RESPONSE_TIME = 'screenTime'
COLUMN_DOMAIN = 'tmpDomain'
COLUMN_TASK_TYPE = 'taskType'

TR = 2.02
N_TR_DEL = 2
GLM_OPTS = dict(drift_model='polynomial', drift_order=1, hrf_model='glover + derivative + dispersion')
#HRF models: 'spm/glover[[ + derivative] + dispersion']

CONFOUND_REGRESSORS_9plus = ['csf', #'white_matter',
                             'framewise_displacement', '(?:trans|rot)_[xyz]$',
                             'motion_outlier_.+']

CONFOUND_REGRESSORS_3plus = ['csf', #'white_matter',
                             'framewise_displacement', 'motion_outlier_.+']

CONFOUND_REGRESSORS = CONFOUND_REGRESSORS_9plus

REF_91K_DENSE = r'C:\Projects\Assuta\Subjects-EKerem\DtSeries\EKW-mental-1-cifti_sbj_Atlas_s0.dtseries.nii'
CIFTI_91K_AXIS = nb.load(REF_91K_DENSE).header.get_axis(1)


def readJson(file):
    metadata = json.load(open(file))
    startTime = metadata['StartTime']
    TR = metadata['RepetitionTime']
    return startTime, TR


def getTR(file):
    df = pd.read_csv(file, header=2, sep='\t')
    pulses = df.loc[df['Event Type']=='Pulse']
    estTR = np.median(np.diff( pulses.astype(float) ))
    return estTR 
    

def readLog():
    parsedLogsFile = opj(BASE_DIR, PARSED_LOGS_FILE_NAME)
    df = pd.read_csv(parsedLogsFile, encoding='utf16')
    
    trialTypes = [(dmn if task=='mor' else 'lex') for dmn, task in df[[COLUMN_DOMAIN, COLUMN_TASK_TYPE]].values ]
    df = df.assign(trial_type=trialTypes)
    df = df.rename(columns={'Time':'onset', COLUMN_RESPONSE_TIME:'duration'})
    
    return df


def getConfoundRegressors(cfdfile, nDel=0, useRegressors=CONFOUND_REGRESSORS):
    df = pd.read_csv(cfdfile, sep='\t').fillna(0)
    allCols = set(df.columns)
    chosenCols = []
    
    for cfdName in useRegressors:
        isReg = '|' in cfdName or '+' in cfdName
        if isReg:
            reg = re.compile(cfdName)
            chosenCols += [col for col in allCols if reg.match(col) ]
            pass
        elif cfdName in allCols:
            chosenCols.append(cfdName)
        else:
            assert f"Couldn't find regressor {cfdName}! Exitting..."         
    
    nTr = len(df)
    motionOutliersCols = [col for col in chosenCols if col.startswith('motion_outlier_')]
    for nd in range(nDel):
        rg = np.zeros(nTr)
        rg[nd] = 1
        alreadeyExists = any(np.allclose(rg, df[col]) for col in motionOutliersCols)
        if not alreadeyExists:
            df = df.assign(**{f'start{nd}':rg})
        
    df = df[chosenCols]
    return df


def createDesignMatrices(sbj, df=None, fwhm=5, taskdir='', jsondir='', regdir='', **kwargs):
    if df is None:
        df = readLog()
        
    if not taskdir:
        taskdir = opj(BASE_DIR,  'Final')
    if not regdir:
        regdir = opj(BASE_DIR,  'Regressors')
    
        
    fwhmSuff = f'_fwhm{fwhm}' if fwhm>0 else ''
    taskfiles = sorted(glob(opj(taskdir, f'{sbj}-mental-*s0{fwhmSuff}.dtseries.nii')))
    cnfdFiles = sorted(glob( opj(regdir, f'sub-{sbj}_*-mental_*') ))
    if not taskfiles or len(taskfiles)!=len(cnfdFiles):
        return None

    columns = ['trial_type', 'onset', 'duration']
    iruns = [int(file.split('\\')[-1].split('-')[2]) for file in taskfiles]
    
    designMatrices = []
    tr = TR
    nDel = N_TR_DEL
    hrfOpts = GLM_OPTS

    for irun, taskfile, cfdfile in zip(iruns, taskfiles, cnfdFiles):
        jsonfile = taskfile.split('.')[0] + '.json'
        jsonfile = glob(opj(jsondir, f'sub-{sbj}_*_run-{iruns}_*-smoothAROMAnonaggr_bold.json'))[0]
        startTime, tr = readJson(jsonfile)
        dfmat = df.query('Subject==@sbj and RunNum==@irun')[columns]

        nTr = nb.load(taskfile).shape[0]
        frameTimes = np.arange(nTr)*tr + startTime
        
        confoundEvs = getConfoundRegressors(cfdfile, nDel)   
        
        design = make_first_level_design_matrix(frameTimes, dfmat, add_regs=confoundEvs, **hrfOpts)
        designMatrices.append( design )

    return designMatrices


def getBoldData(sbj, irun, fwhm=5, bolddir=BOLD_DIR, cortexOnly=True):
    fwhmSuff = f'_fwhm{fwhm}' if fwhm>0 else ''
    boldfile = opj(bolddir, f'{sbj}-mental-{irun}-cifti_sbj_Atlas_s0{fwhmSuff}.dtseries.nii')
    #if not op.exists(boldfile):
    #    return None
    
    niiobj = nb.load(boldfile)
    bms = list(niiobj.header.matrix[1].brain_models)
    n = bms[0].index_count +  bms[1].index_count
    data = niiobj.get_fdata()
    if cortexOnly:
        data = data[:,:n]
    return data


def getContrastVector(design, contrastDetails=None):
    if not contrastDetails:
        contrastDetails = {'space':0, 'time':0, 'person':1, 'lex':-1}
        
    contrastVector = np.zeros(design.shape[1])
    for pref, w in contrastDetails.items():
        for i, col in enumerate(design.columns.values):
            if col==pref:
                contrastVector[i] = w
                
    return contrastVector


def calcTwoContrastsForSubject(sbj, **kwargs):
    designMatrices = createDesignMatrices(sbj, **kwargs)
    designMatrices = {i+1:d for i,d in enumerate(designMatrices)}
    kwargs['designMatrices'] = designMatrices
    
    lexRuns, morRuns = [], []
    for k, desmat in designMatrices.items():
        if 'lex' in desmat.columns:
            lexRuns.append( k )
        else:
            morRuns.append( k )        
            
    shuffle(lexRuns); shuffle(morRuns)    
    nhLex, nhMor = int(len(lexRuns)/2), int(len(morRuns)/2)
    
    useRuns = lexRuns[:nhLex] + morRuns[:nhMor]
    kwargs['fileSuff'] = 'H1'
    calcContrastsForSubject(sbj, useRuns=useRuns, **kwargs)

    kwargs['fileSuff'] = 'H2'
    useRuns = lexRuns[nhLex:] + morRuns[nhMor:]
    calcContrastsForSubject(sbj, useRuns=useRuns, **kwargs)


def duplicate_pad(regressor_vector, run_index, run_boundaries):
    """
    Create a full-length vector with regressor values at specified run indices and zeros elsewhere.
    
    Args:
        regressor_vector: 1D numpy array with regressor values for the specified run
        run_index: Index of the run (0-based) to place the regressor
        run_boundaries: Array of run boundaries [i0, i1, i2, ..., in] where i0=0, 
                       and each boundary indicates the start index of the next run
    
    Returns:
        Full-length vector with regressor at run positions, zeros elsewhere
    """
    run_boundaries = np.asarray(run_boundaries)
    total_length = run_boundaries[-1] if len(run_boundaries) > 0 else len(regressor_vector)
    
    result = np.zeros(total_length)
    
    if run_index < len(run_boundaries) - 1:
        start_idx = run_boundaries[run_index]
        end_idx = run_boundaries[run_index + 1]
        result[start_idx:end_idx] = regressor_vector
    
    return result


def _combine_design_matrices_refactored(
    design_matrices, condition_names, run_boundaries, stack_motion_confounds=True, fix2ndOrderPredictors=False):
    """
    Refactored version of _combine_design_matrices using concat approach.
    
    Args:
        design_matrices: List or dict of pandas DataFrames, one per run
        condition_names: List of condition column names to stack
        run_boundaries: Array of run boundaries [i0, i1, i2, ..., in]
        stack_motion_confounds: If True, stack motion/confound columns; if False, duplicate them
        fix2ndOrderPredictors: If True, columns starting with predictor name + '_' are fixed;
                              If False, only exact matches to predictor names are fixed
    
    Returns:
        combined_design: pandas DataFrame with combined design matrix
    """
    # Convert dict to list if needed, maintaining order
    if isinstance(design_matrices, dict):
        run_indices = sorted(design_matrices.keys())
        design_dict = design_matrices
    else:
        run_indices = list(range(len(design_matrices)))
        design_dict = {i: design_matrices[i] for i in run_indices}
    
    # Build regex pattern for motion/confound columns
    motion_confound_pattern = r'\b(rot|trans|framewise|csf|white)\b'
    motion_confound_regex = re.compile(motion_confound_pattern)
    
    def isColFixed(col):
        """Determine if a column should be fixed (stacked) or duplicated (renamed per run)."""
        col_normalized = col.replace('_', ' ')
        
        # Check if it's a motion/confound column
        if motion_confound_regex.search(col_normalized):
            return stack_motion_confounds
        
        # Check predictor matching
        if fix2ndOrderPredictors:
            # Fixed if column starts with any predictor name followed by '_'
            for pred_name in condition_names:
                if col.startswith(pred_name + '_'):
                    return True
        else:
            # Fixed only if column exactly equals a predictor name
            if col in condition_names:
                return True
        
        # All other columns are duplicated (not fixed)
        return False

    def renameColumns(df, irun):
        """Rename columns that should be duplicated (not fixed) with run suffix."""
        colRenames = {col: f'{col}_{irun}' for col in df.columns if not isColFixed(col)}
        df = df.rename(columns=colRenames)
        return df

    # Rename columns in each design matrix, then concatenate
    renamed_dfs = [renameColumns(design_dict[irun].copy(), irun) for irun in run_indices]
    designMtxCombined = pd.concat(renamed_dfs, ignore_index=True).fillna(0)
    
    return designMtxCombined



def _combine_design_matrices(design_matrices, condition_names, run_boundaries, stack_motion_confounds=False, fix2ndOrderPredictors=False):
    """
    Combine design matrices across runs.
    
    DEPRECATED: Use _combine_design_matrices_refactored instead.
    This function is kept for backward compatibility but delegates to the refactored version.
    
    Args:
        design_matrices: List or dict of pandas DataFrames, one per run
        condition_names: List of condition column names to stack (e.g., ['space', 'time', 'person', 'lex'])
        run_boundaries: Array of run boundaries [i0, i1, i2, ..., in] where i0=0,
                       indicating start indices of each run in the combined matrix
        stack_motion_confounds: If True, stack motion/confound columns; if False, duplicate them
        fix2ndOrderPredictors: If True, columns starting with predictor name + '_' are fixed;
                              If False, only exact matches to predictor names are fixed
    
    Returns:
        combined_design: pandas DataFrame with combined design matrix
    """
    return _combine_design_matrices_refactored(
        design_matrices, condition_names, run_boundaries, stack_motion_confounds, fix2ndOrderPredictors
    )


def _parse_return_stat(return_stat):
    """
    Parse returnStat parameter to determine which stats to compute.
    
    Args:
        return_stat: String (first char: 'z'/'b'/'s'/'p') or dict mapping suffix->stat
                    e.g., dict(P='p', B='b') means compute PSC and beta for all contrasts
    
    Returns:
        List of (suffix, stat_type) tuples, e.g. [('P', 'p'), ('B', 'b')]
        If string, returns [('', 'z')] or similar
    """
    if isinstance(return_stat, dict):
        # Each entry in dict means: compute this stat for ALL contrasts
        # suffix is the file suffix, stat is the stat type
        result = []
        for suffix, stat in return_stat.items():
            stat_type = str(stat).lower()[0] if stat else 'z'
            result.append((suffix, stat_type))
        return result
    else:
        # String: use first (lowercased) character, no suffix
        stat_type = str(return_stat).lower()[0] if return_stat else 'z'
        return [('', stat_type)]


def _normalize_signal(signal, mean_signal, stat_type):
    """
    Normalize signal based on stat_type.
    
    Args:
        signal: Signal array to normalize (can be beta values or BOLD signal)
        mean_signal: Mean signal for normalization
        stat_type: 'p' (normalize to PSC) or 'b'/'s' (normalize to zscale)
    
    Returns:
        Normalized signal array
    """
    stat_type = stat_type.lower()
    
    if stat_type == 'p':
        # Normalize to percent-signal-change: (signal / mean_signal) * 100
        if mean_signal is None:
            raise ValueError("mean_signal required for percent-signal-change normalization")
        return (signal / mean_signal) * 100
    elif stat_type in ('b', 's'):
        # Normalize to zscale: (signal - mean) / std
        if mean_signal is None:
            # No normalization available, return as-is
            return signal
        std_signal = np.std(signal) if hasattr(signal, '__len__') and len(signal) > 1 else 1.0
        if std_signal == 0:
            return signal
        return (signal - mean_signal) / std_signal
    else:
        return signal


def _extract_stat_from_contrast(contrast_obj, stat_type, mean_signal=None):
    """
    Extract requested statistic from contrast object.
    
    Strategy:
    - For all but 'z': first normalize the signal (to 'psc' if 'p', else to zscale)
    - Then 'b'/'p' take beta values, 's' takes std, 'z' takes z (or beta/std)
    
    Args:
        contrast_obj: nilearn Contrast object
        stat_type: 'z' (z-score), 'b' (beta), 's' (std), 'p' (percent-signal-change)
        mean_signal: Mean signal for normalization (required for 'p' and zscale normalization)
    
    Returns:
        numpy array with requested statistic
    """
    stat_type = stat_type.lower()
    
    if stat_type == 'z':
        # For z: take z-score directly (or beta/std)
        return contrast_obj.z_score()
    
    # For all other stats: normalize first, then extract
    beta = contrast_obj.effect_size()
    std = np.sqrt(contrast_obj.effect_variance())
    
    if stat_type == 'p':
        # Normalize to PSC, then return beta (which is now in PSC units)
        normalized_beta = _normalize_signal(beta, mean_signal, 'p')
        return normalized_beta
    elif stat_type == 'b':
        # Normalize to zscale, then return beta values
        normalized_beta = _normalize_signal(beta, mean_signal, 'b')
        return normalized_beta
    elif stat_type == 's':
        # Normalize to zscale, then return std
        # For std, we normalize the beta first, then return the std of normalized values
        normalized_beta = _normalize_signal(beta, mean_signal, 's')
        # Return std of normalized beta (or original std if normalization didn't change scale)
        return std  # std remains the same after zscale normalization
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}. Must be 'z', 'b', 's', or 'p'.")


def calcContrastsForSubject(sbj, designMatrices=None, saveResults=True, outdir=BASE_DIR, 
                            fileSuff='', contrstsDescription=None, multiContrstsDescription=None,
                            taskdir='', jsondir='', regdir='', useRuns=None, returnStat='z', cortexOnly=False):
    """
    sbj : string
    designMatrices : A list of pandas DataFrames, one for each run. 
        If None (default) then created automatically.
    saveResults = True : Boolean - if False, result maps are returned as numpy arrays.
    outdir = None : Relevant for saveResults==True 
    fileSuff = '': Suffix to add to resulting files names
    contrstsDescription : a {cond_i:val_i} dictionary
    multiContrstsDescription : a dictionary of contrasts for running multiple contrasts on one call.
        Overrides <contrstsDescription> if not None
    taskdir : Source of BOLD filees
    jsondir : Source of json filees
    regdir : Source of confounds filees
    useRuns = None: If not None, only use these runs
    returnStat = 'z': Determines which statistic to return:
        - String: First (lowercased) character determines stat ('z'=z-score, 'b'=beta, 's'=std, 'p'=percent-signal-change)
        - Dict: Maps suffix to stat type, e.g. dict(PSC='s', Z='z')
    """

    print(f'Calculating contrasts for {sbj} with returnStat: {returnStat}')

    if not multiContrstsDescription:
        multiContrstsDescription = {'':contrstsDescription}
    
    # Parse returnStat to determine which stats to compute for ALL contrasts
    stat_configs = _parse_return_stat(returnStat)  # List of (suffix, stat_type) tuples
        
    # Create output files: one file per contrast per stat
    outFiles = {}
    for conname in multiContrstsDescription:
        condir = opj(outdir, conname) if conname else outdir
        for suffix, stat_type in stat_configs:
            # Create unique key: (contrast_name, suffix)
            key = (conname, suffix) if suffix else (conname, stat_type)
            if suffix:
                filename = f'{sbj}-{fileSuff}_{suffix}.dscalar.nii'
            else:
                filename = f'{sbj}-{fileSuff}_{stat_type}.dscalar.nii'
            outFiles[key] = opj(condir, filename) 
    
    print(f'Creating files: {outFiles}')
    
    fullyExists = all(op.exists(file) for file in outFiles.values())
    if fullyExists:
        return

    if designMatrices is None:
        designMatrices = createDesignMatrices(sbj, taskdir=taskdir, jsondir=jsondir, regdir=regdir)
    if not isinstance(designMatrices, dict):
        designMatrices = {i+1:d for i,d in enumerate(designMatrices)}
        
    
    contrastObjs = dd(list)
    bold_data_list = []  # Store BOLD data for normalization
    run_lengths = []  # Store run lengths for computing boundaries
    
    # Extract condition names from contrast descriptions
    condition_names = set()
    for condesc in multiContrstsDescription.values():
        if condesc:
            condition_names.update(condesc.keys())
    condition_names = sorted(list(condition_names))
    
    for irun, design in designMatrices.items():
        if useRuns and irun not in useRuns:
            continue
        
        contrastVectors = {name:getContrastVector(design, contrastDetails=condesc) 
                           for name, condesc in multiContrstsDescription.items()}
        data = getBoldData(sbj, irun, bolddir=taskdir, cortexOnly=cortexOnly)
        if data is None:
            print(f'{sbj}: Missing data for run {irun+1}! Skipping run...')
            continue
        
        # Store BOLD data and run length
        bold_data_list.append(data)
        run_lengths.append(data.shape[0])
        
        labels, estimates = run_glm(data, design.values)  
        
        for name, convec in contrastVectors.items():
            # Check if all output files for this contrast already exist
            all_exist = all(op.exists(outFiles.get((name, suffix if suffix else stat_type), '')) 
                          for suffix, stat_type in stat_configs)
            if all_exist:
                continue
        
            contrastObj = compute_contrast(labels, estimates, convec, contrast_type='t')    
            contrastObjs[name].append( contrastObj )
    
    # Compute mean signal for normalization (needed for 'p', 'b', 's' stats)
    mean_signal = None
    if bold_data_list:
        stacked_bold = np.vstack(bold_data_list)
        mean_signal = np.mean(stacked_bold, axis=0)
    
    Results = {}
    # Compute all requested stats for each contrast
    for name, conobj in contrastObjs.items():
        # For each stat configuration (suffix, stat_type)
        for suffix, stat_type in stat_configs:
            # Extract requested stat from each run's contrast
            run_stats = []
            for co in conobj:
                try:
                    stat_val = _extract_stat_from_contrast(co, stat_type, mean_signal=mean_signal)
                    run_stats.append(stat_val)
                except Exception as e:
                    print(f"Warning: Could not extract {stat_type} stat for {name}. Error: {e}")
                    # Fallback to z-score if extraction fails
                    stat_val = co.z_score()
                    run_stats.append(stat_val)
            
            # Simple mean across runs (placeholder until design matrix combination is implemented)
            # Placeholder: In full implementation, use _combine_design_matrices and run single GLM
            if len(run_stats) > 1:
                sess_stat = np.mean(run_stats, axis=0)
            else:
                sess_stat = run_stats[0]
            
            # Store result with key (contrast_name, suffix)
            key = (name, suffix) if suffix else (name, stat_type)
            Results[key] = dict(sess_stat=sess_stat, stat_type=stat_type, suffix=suffix)
    
    if not saveResults:
        return Results
    
    for key, res in Results.items():
        outfile = outFiles[key]
        os.makedirs(op.dirname(outfile), exist_ok=True)
        saveCifti(res['sess_stat'], outfile)
        
    return 



def calcContrastsForAllSubject(contrstsDescription, outdir=CONTRASTS_DIR, fileSuff='',
                                taskdir='', jsondir='', regdir='', doSplitRuns=False, returnStat='z', cortexOnly=False):
    os.makedirs(outdir, exist_ok=True)
        
    logsDf = readLog()
    subjectsWithLogs = set(logsDf.Subject.values)
        
    params = dict(saveResults=True, outdir=outdir, fileSuff='', contrstsDescription=contrstsDescription, 
                  taskdir=taskdir, jsondir=jsondir, regdir=regdir, returnStat=returnStat, cortexOnly=cortexOnly)
    for sbj in subjectsWithLogs:
        designMatrices = createDesignMatrices(sbj, logsDf, taskdir=taskdir, jsondir=jsondir, regdir=regdir)
        if designMatrices is None:
            print(f'{sbj}: Couldn''t find task and/or confound files! Skipping subject...')
            continue
    
        contrastsFu = calcTwoContrastsForSubject if doSplitRuns else calcContrastsForSubject
        contrastsFu(sbj, designMatrices=designMatrices, **params)


def saveCifti(data, outfile, ax1=CIFTI_91K_AXIS, ax0Title='z-score'):
    ax0 = nb.cifti2.ScalarAxis([ax0Title])
    hdr = nb.cifti2.Cifti2Header.from_axes((ax0, ax1))
    
    padData = np.zeros(CIFTI_91K_AXIS.size)
    padData[:data.size] = data
    
    
    tCiiObj = nb.Cifti2Image(dataobj=padData.reshape(1,-1), header=hdr)
    tCiiObj.to_filename( outfile )


def smoothAllBold(bolddir=BOLD_DIR, fwhm=5, sbj=None, dstdir=BOLD_DIR):
    boldfiles = glob(opj(bolddir, '*cifti_sbj_Atlas_s0.dtseries.nii'))
    for infile in boldfiles:
        srcdir, srcfile  = op.split(infile)
        
        parts = srcfile.split('.')
        outfile = opj(dstdir, parts[0]+f'_fwhm{fwhm}.'+'.'.join(parts[1:]))
        
        if op.exists(outfile):
            print(f'Skipping: {infile}')
            continue
                
        print(f'Smoothing: {infile}')
        leftSurface = r'C:\Projects\Templates\S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
        rightSurface = r'C:\Projects\Templates\S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
        
        cmd = ['wb_command', '-cifti-smoothing', infile, str(fwhm), str(fwhm), 'COLUMN', 
               outfile, '-fwhm', '-left-surface', leftSurface, '-right-surface', rightSurface]

        proc = subprocess.Popen(cmd)
        proc.wait()


DEFAULT_CONTRASTS = dict(Space_VS_Lex={'space':1, 'time':0, 'person':0, 'lex':-1},
                         Time_VS_Lex={'space':0, 'time':1, 'person':0, 'lex':-1},
                         Person_VS_Lex={'space':0, 'time':0, 'person':1, 'lex':-1},
                         Space_Time_VS_2Lex={'space':1, 'time':1, 'person':0, 'lex':-2},
                         Space_Time_VS_2Person={'space':1, 'time':1, 'person':-2, 'lex':0},
                         )

if __name__=='__main__':
    baseDir = r'C:\Projects\Assuta\NO_BBR'
    boldDir = opj(baseDir, 'Task_FWHM5')
    if 0:
        rawBoldDir = smoothBoldDir = ''
        smoothAllBold(bolddir=rawBoldDir, dstdir=smoothBoldDir)
    if 0:
        sbjs = ['DaKr', 'ArSh', 'AlZi', 'YiHa']
        #for sbj in sbjs:
            #contrastObjs = calcBetas(sbj)
    
    if 1:
        contrastKeys = 'Space_VS_Lex'.split()
        contrastDir = opj(baseDir, 'Contrasts')
        #doSplitRuns = True

        args = dict(
            taskdir = boldDir,
            jsondir = opj(baseDir, 'Jsons'),
            regdir = opj(baseDir, 'Confounds'),
            returnStat = dict(P='p', Z='z', B='b')
            )
        for name in contrastKeys:
            con = DEFAULT_CONTRASTS[name]
            print(f'Calculating for contrast: {name} - {con}')
            calcContrastsForAllSubject(con, outdir=opj(contrastDir, name), **args)
    if 0:
        # TODO: Get subjects
        # TODO: args
        subjects = []
        args = dict(
            taskdir = '',
            jsondir = '',
            regdir = '',
            )
        designMatrices = {}
        for sbj in subjects:
            designMatrices[sbj] = createDesignMatrices(sbj, **args)
    if 0:
        df = readLog()


