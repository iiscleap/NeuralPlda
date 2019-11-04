#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:52:10 2019

@author: shreyasr
"""

import os
import sys
import numpy as np
import subprocess
import pickle

def kaldifeats2numpydict(inArkOrScpFile, outpicklefile=''):
    #logging.debug('kaldi text file to numpy array: {}'.format(textfile))
    if os.path.splitext(inArkOrScpFile)[1] == '.scp':
        fin = subprocess.check_output(["copy-feats", "scp:{}".format(inArkOrScpFile),"ark,t:-"])
    else: #Assuming ARK
        fin = subprocess.check_output(["copy-feats", "ark:{}".format(inArkOrScpFile),"ark,t:-"])
    res = {}
    fin = fin.decode("utf-8").split('\n')
    while '' in fin:
        fin.remove('')
    tmparr=[]
    arrname=[]
    for line in fin:
        splitted = line.strip().split()
        if splitted[-1] == '[':
            if arrname:
                res[arrname] = np.asarray(tmparr)
            arrname = splitted[0]
            tmparr = []
        else:
            if splitted[-1] == ']':
                splitted = splitted[:-1]
            tmparr.append(np.asarray(splitted).astype(float))
    res[arrname] = np.asarray(tmparr)
    if outpicklefile:
        with open(outpicklefile,'wb') as f:
            pickle.dump(res,f)
    else:
        return res


if __name__=='__main__':
    if len(sys.argv)==1 or len(sys.argv)>=4:
        print("Usage: {} <inArkOrScpFile> <outpicklefile>".format(sys.argv[0]))
    elif len(sys.argv)==2:
        print(kaldifeats2numpydict(sys.argv[1]))
    else:
        kaldifeats2numpydict(sys.argv[1],sys.argv[2])
    