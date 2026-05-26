# -*- coding: utf-8 -*-
import re, sys, yaml, pickle

import numpy as np
import pandas as pd

DIAG_MAP = {'CTRL': 'CU'}


def _predict(df, nullModel, lr):
    params  = nullModel[lr]
    ageOff  = nullModel['Offsets']['Age']
    etivOff = nullModel['Offsets']['eTIV']
    etivCol = nullModel['Columns']['eTIV']

    age_c  = df['DMG_Age'] - ageOff
    etiv_c = df[etivCol]   - etivOff

    return (params['Intercept']
            + params['Age']          * age_c
            + params['Age2']         * age_c**2
            + params['Age3']         * age_c**3
            + params['eTIV']         * etiv_c
            + params.get('eTIV2', 0) * etiv_c**2)


def buildHippoCSVs(sbjDataPkl, configYml, outputPref):
    df        = pickle.load(open(sbjDataPkl, 'rb'))
    cfg       = yaml.safe_load(open(configYml))
    nullModel = cfg['NULL_MODELS']['HippoVol_Null_Model']

    predL = _predict(df, nullModel, 'L')
    predR = _predict(df, nullModel, 'R')

    pat     = re.compile(r'HPVOL_(\w+)_([RL])$')
    sources = sorted({m.group(1) for c in df.columns for m in [pat.match(c)] if m})

    diags = df['Diags'].map(lambda d: DIAG_MAP.get(d, d))

    for src in sources:
        lCol = f'HPVOL_{src}_L'
        rCol = f'HPVOL_{src}_R'
        if lCol not in df.columns or rCol not in df.columns:
            print(f'  SKIP {src}: missing columns')
            continue

        out = pd.DataFrame({
            'SubjectID':     df.index,
            'Age':           df['DMG_Age'].values,
            'Sex':           df['DMG_Sex'].values,
            'Diagnosis':     diags.values,
            'lHipPredicted': predL.values,
            'lHipMeasured':  df[lCol].values,
            'rHipPredicted': predR.values,
            'rHipMeasured':  df[rCol].values,
        })

        outFile = f'{outputPref}_{src}.csv'
        out.to_csv(outFile, index=False)
        print(f'  Saved {len(out)} subjects → {outFile}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python AssutaHippoEst.py <sbjDataPkl> <configYml> <outputPref>')
        sys.exit(1)
    buildHippoCSVs(sys.argv[1], sys.argv[2], sys.argv[3])
