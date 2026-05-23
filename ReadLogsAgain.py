# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:19:32 2026

@author: uriur
ReadLogsAgain
"""

import pandas as pd
import numpy as np

import re

REG_DOMAIN = re.compile(r'doamin_([a-z]+)_pair')    


def makeLog(sbj='OrEr'):
    filenames = ['1mental_orientation_1', '2mental_orientation_2', 
                 '5mental_orientation_3', '6mental_orientation_4']

    tOffset = 0
    dfPics, dfAns, ansCodes = [], [], []
    for name in filenames:
        file = fr'C:\Projects\Assuta\Logs\{sbj}\{name}.log.csv'
        dfpics, dfans, anscodes, Tend = readFile(file, tOffset)
        tOffset = Tend
        dfPics.append( dfpics )
        dfAns.append( dfans )
        ansCodes.append( anscodes )
        
    dfPics = pd.concat(dfPics)
    codes = dfPics.Code.values
    isSame = np.hstack([[True], codes[:-1] != codes[1:]])
    dfPics = dfPics.loc[isSame]
    
    dfAns = pd.concat(dfAns) 
    ansCodes = np.vstack(ansCodes)
    dfPics = getRT(dfPics, dfAns)
    
    return dfPics, dfAns, ansCodes
    

def getRT(dfPics, dfAns):
    iAns = np.searchsorted(dfPics.Time.values, dfAns.Time.values, side='right') - 1
    RT = dfAns.Time.values - dfPics.Time.values[iAns]
    iDups = np.argwhere(np.diff(iAns) == 0).flatten()
    iSkip = np.argwhere(np.diff(iAns) > 1).flatten()
    
    assert len(iSkip)==0, 'Unhandled Case!'
    
    iRemove = []
    for idp in iDups:
        print('Deciding between', RT[idp:idp+2])
        if RT[idp] < 0.1:
            iRemove.append(idp)
        else:
            iRemove.append(idp+1)
            
    v = np.zeros(len(RT))
    v[iRemove] = 1
    iKeep = np.argwhere(v==0).flatten()
    
    dfPics['RT'] = RT[iKeep]
    dfPics['AnsCode'] = dfAns.Code.values[iKeep]
    return dfPics
    
    


def readFile(file, tOffset=0):
    df = pd.read_csv(file, skiprows=3, sep='\t', index_col=0)
    icut = np.argwhere(df.index=='Event Type').flatten()[0]
    df = df.iloc[:icut]
    
    df = df.astype(dict(Time=float, Code=str)).sort_values(by='Time')
    df.Time/= 1e4
    df.Time += tOffset
    df = df.rename(columns = {'Event Type':'EventType'})
    
    Tend = df.Time.values[-1] + 5
    
    def getDomain(s): 
        m = REG_DOMAIN.search(s)
        ans = m.group(1) if m else ''
        return ans
        
    dfPics = df.query('EventType == "Picture"')['Code Time'.split()]
    dfPics['Domain'] = dfPics.Code.apply(getDomain)
    dfPics = dfPics.query('Domain != ""')
    
    dfAns = df.query('EventType == "Response"')['Code Time'.split()]
    c0, c1 = sorted(map(int, dfAns.Code.unique()))
    
    return dfPics, dfAns, (c0, c1), Tend


if __name__ == '__main__':
    dfPics, dfAns, ansCodes = makeLog()


