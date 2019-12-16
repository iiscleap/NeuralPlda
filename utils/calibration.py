#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:58:51 2018

@author: shreyasr
"""


import numpy as np
import scipy.stats as ss
import scipy.optimize as so
from pdb import set_trace as bp

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

def C_norm(thresh, beta, mu_t, mu_nt, sigma_t, sigma_nt):
    return(beta*(1-ss.norm(mu_nt,sigma_nt).cdf(thresh)) #beta*P_{FA}
           + ss.norm(mu_t, sigma_t).cdf(thresh)) #P_Miss

def argmin_cnorm(target_probs, mu_t, mu_nt, sigma_t, sigma_nt):
    betas = [(1-pt)/pt for pt in target_probs]
    Thresholds = []
    for beta in betas:
        def cnorm_instance(x):
            return C_norm(x, beta, mu_t, mu_nt, sigma_t, sigma_nt)
        Thresholds.append((so.minimize_scalar(cnorm_instance)).x)
    return Thresholds

def min_cnorm(target_probs, mu_t, mu_nt, sigma):
    betas = [(1-pt)/pt for pt in target_probs]
    mincs = []
    for beta in betas:
        def cnorm_instance(x):
            return C_norm(x, beta, mu_t, mu_nt, sigma)
        mincs.append(cnorm_instance(so.minimize_scalar(cnorm_instance).x))
    return np.mean(mincs)
        
    
def get_cmn2_thresholds_generative(scores, targets):
    # bp()
    scores_target = scores[targets>0.5]
    scores_nontarget = scores[targets<0.5]
    
    mu_t = np.mean(scores_target)
    mu_nt = np.mean(scores_nontarget)
    
    sigma_t = np.std(scores_target)
    sigma_nt = np.std(scores_nontarget)
    
    minC_threshold1, minC_threshold2 = argmin_cnorm([0.01,0.005],mu_t, mu_nt, sigma_t, sigma_nt)
    min_cent_threshold = minC_threshold2
    
    return minC_threshold1, minC_threshold2, min_cent_threshold


def get_cmn2_thresholds_shared_sigma(scores, targets):
    # bp()
    scores_target = scores[targets>0.5]
    scores_nontarget = scores[targets<0.5]
    
    mu_t = np.mean(scores_target)
    mu_nt = np.mean(scores_nontarget)
    
    sost = np.sum((scores_target-mu_t)**2)
    csc = scores_nontarget-mu_nt
    cscsq = csc**2
    sosnt = np.sum(cscsq)
    sigma = np.sqrt((1/len(scores))*(sost + sosnt))
    
    minC_threshold1, minC_threshold2 = argmin_cnorm([0.01,0.005],mu_t, mu_nt, sigma, sigma)
    min_cent_threshold = minC_threshold2
    
    return minC_threshold1, minC_threshold2, min_cent_threshold


def get_cmn2_thresholds(scores, targets):
    # bp()
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

    inds1 = (scores>bins_cdf[int(min(ind1-1, len(bins_cdf)-1))-1])
    inds2 = (scores<bins_cdf[int(min(ind2+1,len(bins_cdf)-1))])
    inds = inds1 * inds2
    bins_cdf_2 = np.sort(scores[inds])
    
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

