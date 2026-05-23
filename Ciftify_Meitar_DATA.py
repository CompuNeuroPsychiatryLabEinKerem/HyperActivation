# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:24:03 2026

@author: uriur
"""


import nibabel as nb
import os, re, argparse
op = os.path
opj = op.join
from time import time as now
from glob import glob
import pickle
import numpy as np
import subprocess
readPkl = lambda file: pickle.load(open(file, 'rb'))

TR = 2

PROJ_MAIN = '/sci/labs/shahar.arzy/uri.elias/Projects_Data/Assuta_Meitar'
CIFTIFY_MAIN = opj(PROJ_MAIN, 'ciftify', 'ciftify')
FPREP_MAIN = opj(PROJ_MAIN, 'aroma_out')
TEMP_DIR = opj(PROJ_MAIN, 'Temp')
DTSER_DIR = opj(PROJ_MAIN, 'DtSeries')
os.makedirs(TEMP_DIR, exist_ok=True)

ATLAS_ROIS = opj(CIFTIFY_MAIN, 'zz_templates', 'Atlas_ROIs_LPI.2.nii.gz')
SBJ_SURF_DIR = lambda sbj: opj(CIFTIFY_MAIN, f'sub-{sbj}', 'MNINonLinear', 'fsaverage_LR32k')
MED_WALL_FILE = lambda sbj, hm: opj(SBJ_SURF_DIR(sbj), f'sub-{sbj}.{hm}.atlasroi.32k_fs_LR.shape.gii')

def runCmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'FAILED: {cmd}')
        print(f"Error message: \n{result.stderr}")
        
        
def surf_file(sbj, mesh, hemi):
    file = opj(SBJ_SURF_DIR(sbj), f'sub-{sbj}.{hemi}.{mesh}.32k_fs_LR.surf.gii')
    return file


def getFileID(boldfilename):
    baseName = op.split(boldfilename)[-1].split('.')[0]
    shortName = '_'.join(baseName.split('_')[:4])
    return shortName

        
def extractHemi(sbj, volumeFile, hemi):    
    baseName = getFileID(volumeFile)
    output_func = opj(TEMP_DIR, f'temp-{baseName}-{hemi}.func.gii')
    
    cmd = ['wb_command', '-volume-to-surface-mapping', 
           volumeFile, surf_file(sbj, 'midthickness', hemi),
           output_func,'-ribbon-constrained',
           surf_file(sbj, 'white', hemi), surf_file(sbj, 'pial', hemi)]
    
    runCmd(cmd)
    return output_func


def extractSubcortex(sbj, volumeFile):
    fileId = getFileID(volumeFile)
    tmp_fmri_cifti = opj(TEMP_DIR, f'temp_{fileId}_dilate.dtseries.nii')
    tmp_roi_dlabel = opj(TEMP_DIR, f'temp_{fileId}_template.dlabel.nii')
    tmp_atlas_cifti = opj(TEMP_DIR, f'temp_{fileId}_atlas.dtseries.nii')
    output_subcortical = opj(TEMP_DIR, f'Subcortical_{fileId}.nii.gz')
    
    roiFileSbjLas = opj(CIFTIFY_MAIN, f'sub-{sbj}', 'MNINonLinear', 'ROIs', 'ROIs_LPI.2.nii.gz')
    if not op.exists(roiFileSbjLas):
        roiFileSbjRas = opj(CIFTIFY_MAIN, f'sub-{sbj}', 'MNINonLinear', 'ROIs', 'ROIs.2.nii.gz')
        cmd = ['wb_command', '-volume-reorient', roiFileSbjRas, 'LPI', roiFileSbjLas]
        runCmd(cmd)
    
    cmd = ['wb_command', '-cifti-create-dense-timeseries', tmp_fmri_cifti, '-volume', volumeFile, roiFileSbjLas]
    runCmd(cmd)    

    cmd = ['wb_command', '-cifti-create-label', tmp_roi_dlabel, '-volume', ATLAS_ROIS, ATLAS_ROIS]
    runCmd(cmd)

    cmd = ['wb_command', '-cifti-resample', tmp_fmri_cifti, 'COLUMN', tmp_roi_dlabel,
            'COLUMN', 'ADAP_BARY_AREA', 'CUBIC', tmp_atlas_cifti, '-volume-predilate', '10']  
    runCmd(cmd)

    cmd = ['wb_command', '-cifti-separate', tmp_atlas_cifti, 'COLUMN', '-volume-all', output_subcortical]
    runCmd(cmd)
    return output_subcortical


def ciftifyFmri(sbj, bold4dfile):
    shortName = getFileID(bold4dfile)
    tmpFuncFiles = {hemi:extractHemi(sbj, bold4dfile, hemi) for hemi in 'LR'}
    SubCtxFile = extractSubcortex(sbj, bold4dfile)
    ciftiOutput = opj(DTSER_DIR, f'{shortName}_Atlas_s0.dtseries.nii')
    
    cmd = ['wb_command', '-cifti-create-dense-timeseries', ciftiOutput, '-volume', 
           SubCtxFile, ATLAS_ROIS, 
           '-left-metric', tmpFuncFiles['L'], '-roi-left', MED_WALL_FILE(sbj, 'L'), 
           '-right-metric', tmpFuncFiles['R'], '-roi-right', MED_WALL_FILE(sbj, 'R'), 
           '-timestep', f'{TR}']
    runCmd(cmd)    
    
    
def getAllSubjects():
    regSbjDir = re.compile(r'sub-([a-zA-Z]+)/?\Z')
    subjects = sorted(regSbjDir.match(file).group(1) 
                     for file in os.listdir(FPREP_MAIN) if regSbjDir.match(file))
    return subjects
    

def ciftifySubjects(sbj):
    boldsfx = 'nonaggrDenoised_bold.nii.gz'
    boldPattern = lambda sbj: opj(FPREP_MAIN, f'sub-{sbj}', '*', 'func', f'sub-{sbj}*-{boldsfx}')
    bold4dFiles = sorted(glob(boldPattern(sbj)))
    nfiles = len(bold4dFiles)
    for ifile, bold4dfile in enumerate(bold4dFiles):
        print(f'INFO: \t\tProcessing file {ifile}/{nfiles}')
        ciftifyFmri(sbj, bold4dfile)

    
def ciftifyCohort():
    subjects = getAllSubjects()
    nsbjs = len(subjects)
    for isbj, sbj in enumerate(subjects):
        print(f'INFO: {isbj:02}/{nsbjs:02}-{sbj}:  Starting processing...')
        ciftifySubjects(sbj)
            
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', help='Subject ID, e.g. DoGrA')
    parser.add_argument('--array-index', type=int,
                        help='Pick subject by index into sorted list (for SLURM arrays)')
    args = parser.parse_args()
        
    subjects = getAllSubjects()    
    if args.array_index is not None:
        nsbjs = len(subjects)
        i = args.array_index
        sbj = subjects[i] if i<nsbjs else ''
    elif args.subject:
        sbj = args.subject
    else:
        sbj = ''

    if sbj in subjects:
        ciftifySubjects(sbj)



"""
/Projects_Data/Assuta_Meitar/ciftify/ciftify/zz_templates/Atlas_ROIs.2.nii.gz

/Projects_Data/Assuta_Meitar/aroma_out/sub-DoGrA/ses-001/func/sub-DoGrA_ses-001_task-mental_run-001_space-MNI152NLin6Asym_res-2_desc-nonaggrDenoised_bold.nii.gz


cd /Projects_Data/Assuta_Meitar/ciftify/ciftify/zz_templates
srun zsh -c 'wb_command -volume-reorient Atlas_ROIs.2.nii.gz LAS Atlas_ROIs_LAS.2.nii.gz'

srun zsh -c 'wb_command -file-stats Atlas_ROIs_LAS.2.nii.gz'


ciftify/ciftify/sub-DoGrA/MNINonLinear/ROIs/ROIs.2.nii.gz:  -2.000000 0.000000 0.000000 90.000000


"""
