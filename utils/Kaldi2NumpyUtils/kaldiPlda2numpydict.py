#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:17:57 2019

@author: shreyasr
"""

import os
import sys
import numpy as np
import subprocess
import pickle

def kaldiPlda2numpydict(pldaFile, outpicklefile=''):
    #logging.debug('kaldi text file to numpy array: {}'.format(textfile))
    fin = subprocess.check_output(["ivector-copy-plda", "--binary=false", pldaFile ,"-"])
    res = {}
    fin = fin.decode("utf-8").split('\n')
    while '' in fin:
        fin.remove('')
    splitted = fin[0].strip().split()
    res['plda_mean'] = np.asarray(splitted[2:-1]).astype(float)
    tmparr=[]
    for i,line in enumerate(fin[2:]):
        splitted = line.strip().split()
        if splitted[-1] == ']':
            splitted = splitted[:-1]
            tmparr.append(np.asarray(splitted).astype(float))
            break
        else:
            tmparr.append(np.asarray(splitted).astype(float))
    res['diagonalizing_transform'] = np.asarray(tmparr)
    res['Psi_across_covar_diag'] = np.asarray(fin[-2].strip().split()[1:-1]).astype(float)
    ac = res['Psi_across_covar_diag']
    tot = 1 + res['Psi_across_covar_diag']
    res['diagP'] = ac/(tot*(tot-ac*ac/tot))
    res['diagQ'] = (1/tot) - 1/(tot - ac*ac/tot)
    if outpicklefile:
        with open(outpicklefile,'wb') as f:
            pickle.dump(res,f)
    else:
        return res


if __name__=='__main__':
    if len(sys.argv)==1 or len(sys.argv)>=4:
        print("Usage: {} <inArkOrScpFile> <outpicklefile>".format(sys.argv[0]))
    elif len(sys.argv)==2:
        print(kaldiPlda2numpydict(sys.argv[1]))
    else:
        kaldiPlda2numpydict(sys.argv[1],sys.argv[2])
    