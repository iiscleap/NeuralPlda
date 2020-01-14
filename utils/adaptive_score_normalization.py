#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:11:41 2020

@author: shreyasr
"""

import numpy as np

ASnorm_topN = 200

raw_score_filename = ''
cohort_score_filename = ''


raw_scorefile_tab = np.genfromtxt(raw_score_filename, dtype='str')
header = raw_scorefile_tab[0]
raw_scorefile_tab = raw_scorefile_tab[1:]

trials_enroll, trials_test, raw_scores = raw_scorefile_tab[:,0], raw_scorefile_tab[:,1], (raw_scorefile_tab[:,-1]).astype(float)

cohort_scorefile_tab = np.genfromtxt(cohort_score_filename, dtype='str')
cohort_scorefile_tab = cohort_scorefile_tab[1:]
unique_enrolls = np.unique(trials_enroll)
unique_test = np.unique(trials_test)

Se = {enr:cohort_scorefile_tab[:,-1][cohort_scorefile_tab[:,0]==enr] for enr in unique_enrolls}
St = {tst:cohort_scorefile_tab[:,-1][cohort_scorefile_tab[:,0]==tst] for tst in unique_test}

mean_e = {enr:np.mean(Se[enr]) for enr in Se}
std_e = {enr:np.std(Se[enr]) for enr in Se}

mean_t = {tst:np.mean(St[tst]) for tst in St}
std_t = {tst:np.std(St[tst]) for tst in St}

mean_e_top = {enr:np.mean(np.sort(Se[enr])[:ASnorm_topN]) for enr in Se}
std_e_top = {enr:np.std(np.sort(Se[enr])[:ASnorm_topN]) for enr in Se}

mean_t_top = {tst:np.mean(np.sort(St[tst])[:ASnorm_topN]) for tst in St}
std_t_top = {tst:np.std(np.sort(St[tst])[:ASnorm_topN]) for tst in St}

Znorm_scores, Tnorm_scores, Snorm_scores, ASnorm1_scores = [],[],[],[]

for enr, tst, raw_score in zip(trials_enroll, trials_test, raw_scores):
    znorm_scr = (raw_score - mean_e[enr])/std_e[enr]
    tnorm_scr = (raw_score - mean_t[tst])/std_t[tst]
    snorm_scr = (znorm_scr + tnorm_scr)/2
    asnorm1_scr = ((raw_score - mean_e_top[enr])/std_e_top[enr] + (raw_score - mean_t_top[tst])/std_t_top[tst])/2
    
    Znorm_scores.append(znorm_scr)
    Tnorm_scores.append(tnorm_scr)
    Snorm_scores.append(snorm_scr)
    ASnorm1_scores.append(asnorm1_scr)


Znorm_scores = np.asarray(Znorm_scores).astype(str)
Tnorm_scores = np.asarray(Tnorm_scores).astype(str)
Snorm_scores = np.asarray(Snorm_scores).astype(str)
Asnorm1_scores = np.asarray(ASnorm1_scores).astype(str)

np.savetxt(raw_score_filename+'_znorm.tsv',np.c_[raw_scorefile_tab[:,:-1],Znorm_scores], header='\t'.join(header), fmt='%s', delimiter='\t')
np.savetxt(raw_score_filename+'_tnorm.tsv',np.c_[raw_scorefile_tab[:,:-1],Tnorm_scores], header='\t'.join(header), fmt='%s', delimiter='\t')
np.savetxt(raw_score_filename+'_snorm.tsv',np.c_[raw_scorefile_tab[:,:-1],Snorm_scores], header='\t'.join(header), fmt='%s', delimiter='\t')
np.savetxt(raw_score_filename+'_asnorm1.tsv',np.c_[raw_scorefile_tab[:,:-1],ASnorm1_scores], header='\t'.join(header), fmt='%s', delimiter='\t')