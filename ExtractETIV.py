# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:18:05 2026

@author: uriur

ExtractETIV
"""

import numpy as np # noqa
import nibabel as nb # noqa
import pandas as pd # noqa
from scipy.stats import norm
from sklearn.decomposition import PCA
import hcp_utils as hcp
from time import time as now

import os, pickle, re, sys # noqa
op = os.path
opj = op.join
readPkl = lambda file: pickle.load(open(file, 'rb'))
writePkl = lambda obj, file: pickle.dump(obj, open(file, 'wb'))

LOCAL_DIR = op.dirname( op.abspath(__file__) )
FS_DIR = opj(LOCAL_DIR, 'FS_Files')

REG_ASEG_FILE = re.compile(r'(\w{4})-aseg.stats')
REG_ETIV = re.compile(r'Estimated Total Intracranial Volume, (\d+)\.\d+, mm\^3')

MAX_SIZE = 3000
DEFULT_SIZE = 1500

def readAllETIV():
    sbj2etiv = {}
    for f in os.listdir(FS_DIR):
        m = re.match(REG_ASEG_FILE, f)
        if not m:
            continue
        sbj = m.group(1)
        
        txt = open( opj(FS_DIR, f) ).read()
        etiv = float( REG_ETIV.search(txt).group(1) ) / 1e3
        if etiv > MAX_SIZE:
            etiv = DEFULT_SIZE
        
        sbj2etiv[sbj] = etiv
        
    return sbj2etiv
        

if __name__ == '__main__':
    sbj2etiv = readAllETIV()
