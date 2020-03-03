#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:17:04 2020

@author: shreyasr
"""

import numpy as np
import scipy.stats as P
import os
from pdb import set_trace as bp

def calibrate_train(train_scores, train_labels):
    scores_target = train_scores[(train_labels=='target') + (train_labels=='tgt')]
    scores_non_target = train_scores[(train_labels=='nontarget') + (train_labels=='imp')]
    t_mean = np.mean(scores_target)
    nt_mean = np.mean(scores_non_target)
    t_std = np.std(scores_target)
    nt_std = np.std(scores_non_target)
    calib_mdl = dict()
    calib_mdl['tgt'] = P.norm(t_mean, t_std)
    calib_mdl['imp'] = P.norm(nt_mean, nt_std)
    return calib_mdl

def calibrate_apply(scores, calib_mdl):
    scores_calibrated = calib_mdl['tgt'].logpdf(scores) - calib_mdl['imp'].logpdf(scores)
    return scores_calibrated

dev_score_file = '/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/scores/scores_sre18_dev_kaldiplda_xvectors_swbd_sre_mx6_before_norm.tsv'
dev_key_file = '/home/data/SRE2019/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv'

dev_score_files = ['/home/data1/prachis/SRE_19/Focal/fusionsre19_BG/Fusedfinal_sre18_eval_test_score_asnorm1.tsv']
score_files_list = ['/home/data1/prachis/SRE_19/Focal/fusionsre19_BG/Fusedfinal_sre19_eval_test_score_asnorm1.tsv'] #, #dev #Eval score files, cohort score files, etc.

float_formatter = "{:.5f}".format           
np.set_printoptions(formatter={'float_kind':float_formatter})

for f in range(len(score_files_list)):
    scores = np.genfromtxt(dev_score_files[f], dtype='str',
                       skip_header=1)[:,-1].astype(float)
    dev_key = np.genfromtxt(dev_key_file, dtype='str', skip_header=1)[:,3]

    calib_mdl = calibrate_train(scores, dev_key)
    score_tsv = np.genfromtxt(score_files_list[f], dtype='str',skip_header=0)
    scores = score_tsv[1:,-1].astype(float) #or score_tsv[:,-1].astype(float) ## (if there is no header)
    scores_calibrated = calibrate_apply(scores, calib_mdl)
    scores_calibrated_1 = ['{:f}'.format(item) for item in scores_calibrated]
    score_tsv[1:,-1] = scores_calibrated_1#.astype("%.5f")
    save_filename = '_calibrated'.join(os.path.splitext(score_files_list[f]))
    np.savetxt(save_filename, score_tsv, fmt='%s', delimiter='\t',comments='')
