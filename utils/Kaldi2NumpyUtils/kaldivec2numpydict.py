#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:24:18 2019

@author: shreyasr
"""
import os
import sys
import numpy as np
import subprocess
import pickle

def kaldivec2numpydict(inArkOrScpFile, outpicklefile=''):
    #logging.debug('kaldi text file to numpy array: {}'.format(textfile))
    if os.path.splitext(inArkOrScpFile)[1] == '.scp':
        fin = subprocess.check_output(["copy-vector", "scp:{}".format(inArkOrScpFile),"ark,t:-"])
    else: #Assuming ARK
        fin = subprocess.check_output(["copy-vector", "ark:{}".format(inArkOrScpFile),"ark,t:-"])
    res = {}
    fin = fin.decode("utf-8").split('\n')
    while '' in fin:
        fin.remove('')
    for line in fin:
        splitted = line.strip().split()
        res[splitted[0]] = np.asarray(splitted[2:-1]).astype(float)
    if outpicklefile:
        with open(outpicklefile,'wb') as f:
            pickle.dump(res,f)
    else:
        return res


if __name__=='__main__':
    if len(sys.argv)==1 or len(sys.argv)>=4:
        print("Usage: {} <inArkOrScpFile> <outpicklefile>".format(sys.argv[0]))
    elif len(sys.argv)==2:
        print(kaldivec2numpydict(sys.argv[1]))
    else:
        kaldivec2numpydict(sys.argv[1],sys.argv[2])
    