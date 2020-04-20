#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:26:30 2020

@author: shreyasr
"""

import re
import numpy as np
import random
import sys
import subprocess
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
import kaldi_io
from pdb import set_trace as bp
from utils.sv_trials_loaders import load_xvec_trials_from_idbatch

def generate_sre_scores(score_filename, trials_file, mega_dict, model, device, batch_size = 102400):
    # To reduce memory usage on CPU, scores are generated in batches and then concatenated

    model = model.to(torch.device('cpu'))
    trials = np.genfromtxt(trials_file, dtype='str')
    header = '\t'.join(trials[0]) + '\tLLR'
    trials = trials[1:]
    iters = len(trials) // batch_size
    S = torch.tensor([])
    model = model.eval()
    with torch.no_grad():
        for i in range(iters+1):
            x1_b, x2_b = load_xvec_trials_from_idbatch(mega_dict, trials[i * batch_size:i * batch_size + batch_size], device=torch.device('cpu'))
            S_b = model.forward(x1_b, x2_b)
            S = torch.cat((S, S_b))
        scores = np.asarray(S.detach()).astype(str)
    np.savetxt(score_filename, np.c_[trials, scores], header=header, fmt='%s', delimiter='\t', comments='')
    model = model.to(device)
    
def generate_voices_scores(score_filename, trials_file, mega_dict, model, device, batch_size = 102400):
    # To reduce memory usage on CPU, scores are generated in batches and then concatenated

    model = model.to(torch.device('cpu'))
    trials = np.genfromtxt(trials_file, dtype='str')[:,:2]
    iters = len(trials) // batch_size
    S = torch.tensor([])
    model = model.eval()
    with torch.no_grad():
        for i in range(iters+1):
            x1_b, x2_b = load_xvec_trials_from_idbatch(mega_dict, trials[i * batch_size:i * batch_size + batch_size], device=torch.device('cpu'))
            S_b = model.forward(x1_b, x2_b)
            S = torch.cat((S, S_b))
        scores = np.asarray(S.detach()).astype(str)
    np.savetxt(score_filename, np.c_[trials, scores], fmt='%s', delimiter='\t', comments='')
    model = model.to(device)