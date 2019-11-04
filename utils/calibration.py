#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:58:51 2018

@author: shreyasr
"""


import numpy as np


def sigmoid(x):
    z = np.exp(-x)
    sigm = 1. / (1. + z)
    return sigm


def arr2val(x,retidx):
    if x.size>0:
        return x[retidx]
    else:
        return 0

"""
In the following function "get_cmn2_thresholds", We return the threshold at 
which the crossentropy is minimized. However, the calculation of crossentropies
in this way is time consuming, so we simply return 0 as the threshold. This is
used only in the validation step to print the minimum crossentropy, and does 
not affect training, unless the model parameters are updated using this. Feel 
free to suggest alternate implementation for speed and memory optimization.
"""
    
def get_cmn2_thresholds(scores, targets):
    scores_target = scores[targets>0.5]
    scores_nontarget = scores[targets<0.5]

    scores_target_sorted = np.sort(scores_target)
    scores_nontarget_sorted_rev_temp = np.sort(-scores_nontarget)
    scores_nontarget_sorted_rev = -scores_nontarget_sorted_rev_temp


    bins_cdf = np.sort(scores)
    stepsize = int(np.ceil(len(bins_cdf)/5000))
    bins_cdf = bins_cdf[0::stepsize]
    
    
    pmiss = np.asarray([arr2val(np.where(scores_target_sorted<i)[0],-1) for i in bins_cdf])/len(scores_target)
    pfa = np.asarray([arr2val(np.where(scores_nontarget_sorted_rev>=i)[0],-1) for i in bins_cdf])/len(scores_nontarget)
#    crossentropies = -(1/len(scores))*np.asarray([np.sum(np.log(sigmoid(scores_target_sorted-i))) + np.sum(np.log(1-sigmoid(scores_nontarget_sorted_rev-i))) for i in bins_cdf])
    
    minC_score_idx1 = np.argmin(pmiss+99*pfa)
    minC_score_idx2 = np.argmin(pmiss+199*pfa)
    min_cent_idx = minC_score_idx2 #np.argmin(crossentropies)

    ind1 = int(min(minC_score_idx1, minC_score_idx2,min_cent_idx))
    ind2 = int(max(minC_score_idx1, minC_score_idx2,min_cent_idx))
    
    bins_cdf_2 = np.sort(scores[(scores>bins_cdf[int(min(ind1-1, len(bins_cdf)-1))-1]) * (scores<bins_cdf[int(min(ind2+1,len(bins_cdf)-1))])])
    
    pmiss = np.asarray([arr2val(np.where(scores_target_sorted<i)[0],-1) for i in bins_cdf_2])/len(scores_target)
    pfa = np.asarray([arr2val(np.where(scores_nontarget_sorted_rev>=i)[0],-1) for i in bins_cdf_2])/len(scores_nontarget)
#    crossentropies = -(1/len(scores))*np.asarray([np.sum(np.log(sigmoid(scores_target_sorted-i))) + np.sum(np.log(1-sigmoid(scores_nontarget_sorted_rev-i))) for i in bins_cdf_2])
    
    minC_score_idx1 = np.argmin(pmiss+99*pfa)
    minC_score_idx2 = np.argmin(pmiss+199*pfa)
    min_cent_idx = minC_score_idx2 #np.argmin(crossentropies)
    
    minC_threshold1 = bins_cdf_2[minC_score_idx1]
    minC_threshold2 = bins_cdf_2[minC_score_idx2]
    min_cent_threshold = 0.0 #bins_cdf_2[minC_score_idx2]
    
    return minC_threshold1, minC_threshold2, min_cent_threshold
    

def get_vast_thresholds(scores, targets):
    scores_target = scores[targets>0.5]
    scores_nontarget = scores[targets<0.5]

    scores_target_sorted = np.sort(scores_target)
    scores_nontarget_sorted_rev_temp = np.sort(-scores_nontarget)
    scores_nontarget_sorted_rev = -scores_nontarget_sorted_rev_temp


    bins_cdf = np.sort(scores)
    stepsize = int(np.ceil(len(bins_cdf)/5000))
    bins_cdf = bins_cdf[0::stepsize]
    
    
    pmiss = np.asarray([arr2val(np.where(scores_target_sorted<i)[0],-1) for i in bins_cdf])/len(scores_target)
    pfa = np.asarray([arr2val(np.where(scores_nontarget_sorted_rev>=i)[0],-1) for i in bins_cdf])/len(scores_nontarget)
    crossentropies = -(1/len(scores))*np.asarray([np.sum(np.log(sigmoid(scores_target_sorted-i))) + np.sum(np.log(1-sigmoid(scores_nontarget_sorted_rev-i))) for i in bins_cdf])

    
    minC_score_idx = np.argmin(pmiss+19*pfa)
    min_cent_idx = np.argmin(crossentropies)
    
    ind1 = int(min(minC_score_idx, min_cent_idx))
    ind2 = int(max(minC_score_idx, min_cent_idx))
    bins_cdf_2 = np.sort(scores[(scores>bins_cdf[int(min(ind1-1, len(bins_cdf)-1))-1]) * (scores<bins_cdf[int(min(ind2+1,len(bins_cdf)-1))])])
    
    pmiss = np.asarray([arr2val(np.where(scores_target_sorted<i)[0],-1) for i in bins_cdf_2])/len(scores_target)
    pfa = np.asarray([arr2val(np.where(scores_nontarget_sorted_rev>=i)[0],-1) for i in bins_cdf_2])/len(scores_nontarget)
    crossentropies = -(1/len(scores))*np.asarray([np.sum(np.log(sigmoid(scores_target_sorted-i))) + np.sum(np.log(1-sigmoid(scores_nontarget_sorted_rev-i))) for i in bins_cdf_2])
    
    minC_score_idx = np.argmin(pmiss+19*pfa)  
    min_cent_idx = np.argmin(crossentropies)

    minC_threshold = bins_cdf_2[minC_score_idx]
    min_cent_threshold = bins_cdf_2[min_cent_idx]
    return minC_threshold, min_cent_threshold

