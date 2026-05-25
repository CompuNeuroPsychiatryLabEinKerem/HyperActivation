# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd


def buildReport(fcPklFile, demoCSVFile, outputXLSX):
    data     = pickle.load(open(fcPklFile, 'rb'))
    fc       = data['FC']        # (nFiles, 2, nParcels)
    runsDF   = data['Runs']      # Subject, Run, TaskName, OrigID
    labelsDF = data['Labels']    # Label, Hemi, Network, SubParcel, Desc

    demoDF   = pd.read_csv(demoCSVFile)

    # Sanity: subject sets
    runSbjs  = set(runsDF['Subject'])
    demoSbjs = set(demoDF['SubjectID'])
    print(f'Subjects in runs only : {sorted(runSbjs - demoSbjs) or "none"}')
    print(f'Subjects in demo only : {sorted(demoSbjs - runSbjs) or "none"}')
    print(f'Subjects in both      : {sorted(runSbjs & demoSbjs)}')

    # FC columns: zCorr_RspSum_Reg{label} then zCorr_RspAvg_Reg{label}
    labels   = labelsDF['Label'].values
    seedTags = ['RspSum', 'RspAvg']          # matches Seeds order ['sum', 'normMean']
    fcCols   = {}
    for si, tag in enumerate(seedTags):
        for ri, lbl in enumerate(labels):
            fcCols[f'zCorr_{tag}_Reg{lbl}'] = fc[:, si, ri]

    runsFCDF = pd.concat([runsDF.reset_index(drop=True),
                          pd.DataFrame(fcCols)], axis=1)

    mainDF   = runsFCDF.merge(demoDF, left_on='Subject', right_on='SubjectID', how='outer')
    mainDF   = mainDF.drop(columns=['SubjectID'])

    # Column order: run descriptors, demographics, FC
    runCols  = list(runsDF.columns)
    demoCols = [c for c in demoDF.columns if c != 'SubjectID']
    mainDF   = mainDF[runCols + demoCols + list(fcCols)]

    regDF    = labelsDF[['Label', 'Hemi', 'Network', 'SubParcel', 'Desc']].copy()

    with pd.ExcelWriter(outputXLSX, engine='openpyxl') as writer:
        mainDF.to_excel(writer, sheet_name='Runs',    index=False)
        regDF.to_excel( writer, sheet_name='Regions', index=False)

    print(f'Saved {len(mainDF)} rows × {len(mainDF.columns)} cols → {outputXLSX}', flush=True)
    return mainDF, regDF


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Build final report XLSX from FC pickle and demographics.')
    p.add_argument('fcPklFile',   help='FC pickle (from AssutaFC.py)')
    p.add_argument('demoCSVFile', help='Demographics/hippocampi CSV')
    p.add_argument('outputXLSX', help='Output Excel file path')
    args = p.parse_args()
    buildReport(args.fcPklFile, args.demoCSVFile, args.outputXLSX)
