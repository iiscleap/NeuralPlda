#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:58:13 2020

@author: shreyasr
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pickle
import subprocess
from utils.Kaldi2NumpyUtils.kaldiPlda2numpydict import kaldiPlda2numpydict
from pdb import set_trace as bp

def arr2val(x, retidx):
    if x.size()[0] > 0:
        return x[retidx].cpu().item()
    else:
        return 1.

class NeuralPlda(nn.Module):
    def __init__(self, nc):
        super(NeuralPlda, self).__init__()
        self.centering_and_LDA = nn.Linear(nc.xvector_dim, nc.layer1_LDA_dim)  # Centering, wccn
        self.centering_and_wccn_plda = nn.Linear(nc.layer1_LDA_dim, nc.layer2_PLDA_spkfactor_dim)
        self.P_sqrt = nn.Parameter(torch.rand(nc.layer2_PLDA_spkfactor_dim, requires_grad=True))
        self.Q = nn.Parameter(torch.rand(nc.layer2_PLDA_spkfactor_dim, requires_grad=True))
        self.threshold = {}
        for beta in nc.beta:
            self.threshold[beta] = nn.Parameter(0*torch.rand(1, requires_grad=True))
            self.register_parameter("Th{}".format(int(beta)), self.threshold[beta])
        self.threshold_Xent = nn.Parameter(0*torch.rand(1, requires_grad=True))
        self.threshold_Xent.requires_grad = False
        self.alpha = torch.tensor(nc.alpha).to(nc.device)
        self.beta = nc.beta
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x1, x2):
        x1 = self.centering_and_LDA(x1)
        x2 = self.centering_and_LDA(x2)
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)

        x1 = self.centering_and_wccn_plda(x1)
        x2 = self.centering_and_wccn_plda(x2)
        P = self.P_sqrt * self.P_sqrt
        Q = self.Q
        S = (x1 * Q * x1).sum(dim=1) + (x2 * Q * x2).sum(dim=1) + 2 * (x1 * P * x2).sum(dim=1)

        return S

    def softcdet(self, output, target):
        sigmoid = nn.Sigmoid()
        losses = [((sigmoid(self.alpha * (self.threshold[beta] - output)) * target).sum() / (target.sum()) + beta * (sigmoid(self.alpha * (output - self.threshold[beta])) * (1 - target)).sum() / ((1 - target).sum())) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss

    def cdet(self, output, target):
        losses = [((output < self.threshold[beta]).float() * target).sum() / (target.sum()) + beta * ((output > self.threshold[beta]).float() * (1 - target)).sum() / ((1 - target).sum()) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def minc(self, output, target, update_thresholds=False):
        scores_target, _ = torch.sort(output[target>0.5])
        scores_nontarget, _ = torch.sort(-output[target<0.5])
        scores_nontarget = -scores_nontarget
        pmiss_arr = [arr2val(torch.where(scores_target < i)[0], -1) for i in scores_target]
        pmiss = torch.tensor(pmiss_arr) / (target.sum())
        pfa = torch.tensor([arr2val(torch.where(scores_nontarget >= i)[0], -1) for i in scores_target]) / (1-target).sum()
        cdet_arr, minc_dict, minc_threshold = {}, {}, {}
        for beta in self.beta:
            cdet_arr[beta] = pmiss + beta*pfa
            minc_dict[beta], thidx = torch.min(cdet_arr[beta], 0)
            minc_threshold[beta] = scores_target[thidx]
            if update_thresholds:
                self.state_dict()["Th{}".format(int(beta))].data.copy_(minc_threshold[beta])
        mincs = list(minc_dict.values())
        minc_avg = sum(mincs)/len(mincs)
        return minc_avg, minc_threshold
            
        
        

    def LoadPldaParamsFromKaldi(self, mean_vec_file, transform_mat_file, PldaFile):
        plda = kaldiPlda2numpydict(PldaFile)
        transform_mat = np.asarray([w.split() for w in np.asarray(
            subprocess.check_output(["copy-matrix", "--binary=false", transform_mat_file, "-"]).decode('utf-8').strip()[
            2:-2].split('\n'))]).astype(float)
        mean_vec = np.asarray(
            subprocess.check_output(["copy-vector", "--binary=false", mean_vec_file, "-"]).decode('utf-8').strip()[
            1:-2].split()).astype(float)
        mdsd = self.state_dict()
        mdsd['centering_and_LDA.weight'].data.copy_(torch.from_numpy(transform_mat[:, :-1]).float())
        mdsd['centering_and_LDA.bias'].data.copy_(
            torch.from_numpy(transform_mat[:, -1] - transform_mat[:, :-1].dot(mean_vec)).float())
        mdsd['centering_and_wccn_plda.weight'].data.copy_(torch.from_numpy(plda['diagonalizing_transform']).float())
        mdsd['centering_and_wccn_plda.bias'].data.copy_(
            torch.from_numpy(-plda['diagonalizing_transform'].dot(plda['plda_mean'])).float())
        mdsd['P_sqrt'].data.copy_(torch.from_numpy(np.sqrt(plda['diagP'])).float())
        mdsd['Q'].data.copy_(torch.from_numpy(plda['diagQ']).float())

    def SaveModel(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class GaussianBackend(nn.Module): 
    def __init__(self, nc):
        super(GaussianBackend, self).__init__()
        self.centering_and_LDA = nn.Linear(nc.xvector_dim, nc.layer1_LDA_dim)
        self.centering_and_LDA.weight.requires_grad = False
        self.centering_and_LDA.bias.requires_grad = False
        self.paired_mean_target = torch.rand(2*nc.layer1_LDA_dim)
        self.paired_cov_inv_target = torch.rand(2*nc.layer1_LDA_dim,2*nc.layer1_LDA_dim)
        self.paired_mean_nontarget = torch.rand(2*nc.layer1_LDA_dim)
        self.paired_cov_inv_nontarget = torch.rand(2*nc.layer1_LDA_dim,2*nc.layer1_LDA_dim)

        
		
    def forward(self, x1, x2):
        x1 = self.centering_and_LDA(x1) #(x1-self.mu)/self.stddev
        x2 = self.centering_and_LDA(x2) #(x2-self.mu)/self.stddev
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x = torch.cat((x1,x2),dim=1)
        St = (-((x-self.paired_mean_target).mm(self.paired_cov_inv_target))*(x-self.paired_mean_target)).sum(dim=1) 
        Snt = (-((x-self.paired_mean_nontarget).mm(self.paired_cov_inv_nontarget))*(x-self.paired_mean_nontarget)).sum(dim=1)
        S = St - Snt
        return S
    
    def forward_getpaired(self, x1, x2):
        x1 = self.centering_and_LDA(x1) #(x1-self.mu)/self.stddev
        x2 = self.centering_and_LDA(x2) #(x2-self.mu)/self.stddev
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x = torch.cat((x1,x2),dim=1)
        return x
    

    def LoadPldaParamsFromKaldi(self, mean_vec_file, transform_mat_file):
        transform_mat = np.asarray([w.split() for w in np.asarray(subprocess.check_output(["copy-matrix","--binary=false", transform_mat_file, "-"]).decode('utf-8').strip()[2:-2].split('\n'))]).astype(float)
        mean_vec = np.asarray(subprocess.check_output(["copy-vector", "--binary=false", mean_vec_file, "-"]).decode('utf-8').strip()[1:-2].split()).astype(float)
        mdsd = self.state_dict()
        mdsd['centering_and_LDA.weight'].data.copy_(torch.from_numpy(transform_mat[:,:-1]).float())
        mdsd['centering_and_LDA.bias'].data.copy_(torch.from_numpy(transform_mat[:,-1]-transform_mat[:,:-1].dot(mean_vec)).float())
        
    def SaveModel(self, filename):
        with open(filename,'wb') as f:
            pickle.dump(self,f)