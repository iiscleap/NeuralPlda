#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:10:11 2020

@author: shreyasr
"""
import numpy as np
from matplotlib import pyplot
import os


score_key = np.genfromtxt('/run/user/1001/gvfs/sftp:host=10.64.18.30,user=prashantk/home/data/SRE2019/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv', dtype='str', delimiter='\t', skip_header=1)

cmn2_target_idx = score_key[:,3]=='target' #(score_key[:,-1]=='cmn2') #* ()
cmn2_nontarget_idx = score_key[:,3]=='nontarget' #(score_key[:,-1]=='cmn2') #* ()

timestamp=1579168538
for i in range(1,31):
    score_tsv = np.genfromtxt('/run/user/1001/gvfs/sftp:host=10.64.18.30,user=prashantk/home/data2/SRE2019/prashantk/NeuralPlda/scores/sre18_dev_kaldipldanet_epoch{}_{}.txt'.format(i,timestamp), dtype='str', delimiter='\t', skip_header=0)
    
    header = score_tsv[0]
    score_tsv = score_tsv[1:]
    scores = (score_tsv[:,-1]).astype(float)
    
    scores_target_cmn2 = scores[cmn2_target_idx]
    scores_nontarget_cmn2 = scores[cmn2_nontarget_idx]
    
    max_scores = max(scores)
    min_scores = min(scores)
    
    bins_cmn2 = np.linspace(min_scores, max_scores, 200)
    bins_vast = np.linspace(min_scores, max_scores, 20)
    
    pyplot.figure()
    pyplot.hist(scores_target_cmn2, bins_cmn2, alpha=0.5, label='target_cmn2')
    pyplot.hist(scores_nontarget_cmn2, bins_cmn2, alpha=0.5, label='nontarget_cmn2')
    pyplot.axis([-2.,2.,0.,2100])
    if not os.path.exists('plots/histograms_{}/'.format(timestamp)):
        os.makedirs('plots/histograms_{}/'.format(timestamp))
    pyplot.savefig('plots/histograms_{}/hist_epoch{}_{}.png'.format(timestamp,i,timestamp))