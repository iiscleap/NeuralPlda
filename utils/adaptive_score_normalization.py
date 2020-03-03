#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:11:41 2020

@author: shreyasr
"""

import numpy as np
# from pdb import set_trace as bp

ASnorm_topN = 500

# raw_score_filename = '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/scores_18/scores_sre18_dev_kaldiplda_xvectors_swbd_sre_mx6_sre16_before_norm.tsv'
# cohort_score_filename = '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/scores_18/scores_sre18_dev_cohort_kaldiplda_xvectors_swbd_sre_mx6_sre16_before_norm.tsv'

raw_score_filename = '/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/scores/scores_sre18_eval_kaldiplda_xvectors_swbd_sre_mx6_before_norm.tsv'
cohort_score_filename = '/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/scores/scores_sre18_eval_cohort_kaldiplda_xvectors_swbd_sre_mx6_before_norm.tsv'

raw_scorefile_tab = np.genfromtxt(raw_score_filename, dtype='str')
header = raw_scorefile_tab[0]
raw_scorefile_tab = raw_scorefile_tab[1:]

trials_enroll, trials_test, raw_scores = raw_scorefile_tab[:,0], raw_scorefile_tab[:,1], (raw_scorefile_tab[:,-1]).astype(float)
trials_test = np.asarray([w.replace('.sph','') for w in trials_test])

cohort_scorefile_tab = np.genfromtxt(cohort_score_filename, dtype='str', skip_header=1)
num_unlabelled = len(np.unique(cohort_scorefile_tab[:,1]))
unique_enrolls = np.unique(trials_enroll)
unique_test = np.unique(trials_test)

cohort_score_matrix = np.sort(cohort_scorefile_tab[:,-1].astype(float).reshape(-1,num_unlabelled),axis=1)
mean_cohort_scores = np.mean(cohort_score_matrix, axis=1)
std_cohort_scores = np.std(cohort_score_matrix, axis=1)
mean_top_cohort_scores = np.mean(cohort_score_matrix[:,:ASnorm_topN], axis=1)
std_top_cohort_scores = np.std(cohort_score_matrix[:,:ASnorm_topN], axis=1)

enrolls_of_cohort = cohort_scorefile_tab[:,0].reshape(-1,num_unlabelled)[:,0]


S = dict(zip(enrolls_of_cohort, cohort_score_matrix))
mean_dict = dict(zip(enrolls_of_cohort, mean_cohort_scores))
std_dict = dict(zip(enrolls_of_cohort, std_cohort_scores))
mean_dict_top = dict(zip(enrolls_of_cohort, mean_top_cohort_scores))
std_dict_top = dict(zip(enrolls_of_cohort, std_top_cohort_scores))


# Se = {enr:cohort_scorefile_tab[:,-1][cohort_scorefile_tab[:,0]==enr].astype(float) for enr in unique_enrolls}
# St = {tst:cohort_scorefile_tab[:,-1][cohort_scorefile_tab[:,0]==tst].astype(float) for tst in unique_test}

# mean_e = {enr:np.mean(Se[enr]) for enr in Se}
# std_dict = {enr:np.std(Se[enr]) for enr in Se}

# mean_dict = {tst:np.mean(St[tst]) for tst in St}
# std_dict = {tst:np.std(St[tst]) for tst in St}

# mean_e_top = {enr:np.mean(np.sort(Se[enr])[:ASnorm_topN]) for enr in Se}
# std_dict_top = {enr:np.std(np.sort(Se[enr])[:ASnorm_topN]) for enr in Se}

# mean_dict_top = {tst:np.mean(np.sort(St[tst])[:ASnorm_topN]) for tst in St}
# std_dict_top = {tst:np.std(np.sort(St[tst])[:ASnorm_topN]) for tst in St}

Znorm_scores, Tnorm_scores, Snorm_scores, ASnorm1_scores = [],[],[],[]

for enr, tst, raw_score in zip(trials_enroll, trials_test, raw_scores):
    znorm_scr = (raw_score - mean_dict[enr])/std_dict[enr]
    tnorm_scr = (raw_score - mean_dict[tst])/std_dict[tst]
    snorm_scr = (znorm_scr + tnorm_scr)/2
    asnorm1_scr = ((raw_score - mean_dict_top[enr])/std_dict_top[enr] + (raw_score - mean_dict_top[tst])/std_dict_top[tst])/2
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