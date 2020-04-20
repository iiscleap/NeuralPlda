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
from matplotlib import pyplot as plt
from pdb import set_trace as bp

def arr2val(x, retidx):
    if x.size()[0] > 0:
        return x[retidx].cpu().item()
    else:
        return 1.

class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=23,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.padlen = int(dilation * (context_size - 1) / 2)
        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        output: size (batch, new_seq_len, output_features)
        '''
        # print("In forward of TDNN")
        batch_size, _, d = tuple(x.shape)
        # print("X : ",x.shape)
        # print("D = ",d)
        # print(self.input_dim)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        # bp()

        x = F.unfold(x, (self.context_size, self.input_dim), stride=(1, self.input_dim), dilation=(self.dilation, 1))

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,
                        2)  # .reshape(-1,self.context_size, self.input_dim).flip(0,1).flip(1,2).flip(1,0,2).reshape(batch_size,-1,self.context_size*self.input_dim)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


class XVectorNet_ETDNN_12Layer(nn.Module):
    def __init__(self, noclasses=13539, pooling_function=torch.std):
        super(XVectorNet_ETDNN_12Layer, self).__init__()
        self.tdnn1 = TDNN(input_dim=30, output_dim=512, context_size=5, dilation=1)
        # self.tdnn1.requires_grad = False
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.tdnn2.requires_grad = False
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        # self.tdnn3.requires_grad = False
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.tdnn4.requires_grad = False
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        # self.tdnn5.requires_grad = False
        self.tdnn6 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.tdnn6.requires_grad = False
        self.tdnn7 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=4)
        # self.tdnn7.requires_grad = False
        self.tdnn8 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.tdnn8.requires_grad = False
        self.tdnn9 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.tdnn9.requires_grad = False
        self.tdnn10 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        # self.tdnn10.requires_grad = False
        self.pooling_function = pooling_function
        self.lin11 = nn.Linear(3000, 512)
        self.bn11 = nn.BatchNorm1d(num_features=512, affine=False)
        self.bn12 = nn.BatchNorm1d(num_features=512, affine=False)
        self.lin12 = nn.Linear(512, 512)
        self.finlin = nn.Linear(512, noclasses)
        self.smax = nn.Softmax(dim=1)
        
        
    
        
        
    def prestatspool(self, x):
        # bp()
        x = F.dropout(self.tdnn1(x), p=0.5)
        x = F.dropout(self.tdnn2(x), p=0.5)
        x = F.dropout(self.tdnn3(x), p=0.5)
        x = F.dropout(self.tdnn4(x), p=0.5)
        x = F.dropout(self.tdnn5(x), p=0.5)
        x = F.dropout(self.tdnn6(x), p=0.5)
        x = F.dropout(self.tdnn7(x), p=0.5)
        x = F.dropout(self.tdnn8(x), p=0.5)
        x = F.dropout(self.tdnn9(x), p=0.5)
        x = F.dropout(self.tdnn10(x), p=0.5)
        return x

    def statspooling(self, x):
        average = x.mean(1)
        stddev = self.pooling_function(x,1) # x.std(1)
        concatd = torch.cat((average, stddev), 1)
        return concatd

    def postpooling(self, x):
        x = F.dropout(self.bn11(F.relu(self.lin11(x))), p=0.5)
        x = F.dropout(self.bn12(F.relu(self.lin12(x))), p=0.5)
        x = F.relu(self.finlin(x))
        return x

    def forward(self, x):
        x = x.transpose(1, 2)
        # bp()
        # print('In forward of XvectorNet')
        prepoolout = self.prestatspool(x)
        pooledout = self.statspooling(prepoolout)
        presoftmax = self.postpooling(pooledout)
        finaloutput = self.smax(presoftmax)
        return finaloutput

    def extract(self, x):
        x = x.transpose(1, 2)
        # x = self.prestatspool(x)
        x = self.tdnn1.forward(x)
        x = self.tdnn2.forward(x)
        x = self.tdnn3.forward(x)
        x = self.tdnn4.forward(x)
        x = self.tdnn5.forward(x)
        x = self.tdnn6.forward(x)
        x = self.tdnn7.forward(x)
        x = self.tdnn8.forward(x)
        x = self.tdnn9.forward(x)
        x = self.tdnn10.forward(x)
        pooledout = self.statspooling(x)
        xvec = self.lin11.forward(pooledout)
        return xvec

    def LoadFromKaldi(self, weightspath):  # Credits: Harsha Varshan
        with open(weightspath, 'rb') as f:
            kaldiweights = pickle.load(f)

        mdsd = self.state_dict()

        for i in range(1, 11):
            mdsd['tdnn{}.kernel.weight'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.affine'.format(i)]['params']).float())
            mdsd['tdnn{}.kernel.bias'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.affine'.format(i)]['bias']).float())
            mdsd['tdnn{}.bn.running_mean'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.batchnorm'.format(i)]['stats-mean']).float())
            mdsd['tdnn{}.bn.running_var'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.batchnorm'.format(i)]['stats-var']).float())

        mdsd['lin11.weight'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.affine']['params']).float())
        mdsd['lin11.bias'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.affine']['bias']).float())
        mdsd['bn11.running_mean'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.batchnorm']['stats-mean']).float())
        mdsd['bn11.running_var'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.batchnorm']['stats-var']).float())

        mdsd['lin12.weight'].data.copy_(torch.from_numpy(kaldiweights['tdnn12.affine']['params']).float())
        mdsd['lin12.bias'].data.copy_(torch.from_numpy(kaldiweights['tdnn12.affine']['bias']).float())
        mdsd['bn12.running_mean'].data.copy_(torch.from_numpy(kaldiweights['tdnn12.batchnorm']['stats-mean']).float())
        mdsd['bn12.running_var'].data.copy_(torch.from_numpy(kaldiweights['tdnn12.batchnorm']['stats-var']).float())

        mdsd['finlin.weight'].data.copy_(torch.from_numpy(kaldiweights['output.affine']['params']).float())
        mdsd['finlin.bias'].data.copy_(torch.from_numpy(kaldiweights['output.affine']['bias']).float())


class Etdnn_Xvec_NeuralPlda(nn.Module):
    def __init__(self, nc):
        super(Etdnn_Xvec_NeuralPlda, self).__init__()
        if nc.pooling_function=='var':
            self.pooling_function = torch.var
        else:
            self.pooling_function = torch.std
        self.xvector_extractor = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
        self.centering_and_LDA = nn.Linear(nc.xvector_dim, nc.layer1_LDA_dim)  # Centering, wccn
        self.centering_and_wccn_plda = nn.Linear(nc.layer1_LDA_dim, nc.layer2_PLDA_spkfactor_dim)
        self.P_sqrt = nn.Parameter(torch.rand(nc.layer2_PLDA_spkfactor_dim, requires_grad=True))
        self.Q = nn.Parameter(torch.rand(nc.layer2_PLDA_spkfactor_dim, requires_grad=True))
        self.threshold = {}
        for beta in nc.beta:
            self.threshold[beta] = nn.Parameter(0*torch.rand(1, requires_grad=True))
            self.register_parameter("Th{}".format(int(beta)), self.threshold[beta])
        self.threshold_Xent = nn.Parameter(0*torch.rand(1, requires_grad=True))
        self.alpha = torch.tensor(nc.alpha).to(nc.device)
        self.beta = nc.beta
        self.dropout = nn.Dropout(p=0.5)
        self.lossfn = nc.loss
        
    def train1(self):
        self.train()
        self.xvector_extractor.tdnn1.bn.training = False
        self.xvector_extractor.tdnn2.bn.training = False
        self.xvector_extractor.tdnn3.bn.training = False
        self.xvector_extractor.tdnn4.bn.training = False
        self.xvector_extractor.tdnn5.bn.training = False
        self.xvector_extractor.tdnn6.bn.training = False
        self.xvector_extractor.tdnn7.bn.training = False
        self.xvector_extractor.tdnn8.bn.training = False
        self.xvector_extractor.tdnn9.bn.training = False
        self.xvector_extractor.tdnn10.bn.training = False
        
    def extract_plda_embeddings(self, x):
        x = self.xvector_extractor.extract(x)
        x = self.centering_and_LDA(x)
        x = F.normalize(x)
        x = self.centering_and_wccn_plda(x)
        return x

    def forward_from_plda_embeddings(self,x1,x2):
        P = self.P_sqrt * self.P_sqrt
        Q = self.Q
        S = (x1 * Q * x1).sum(dim=1) + (x2 * Q * x2).sum(dim=1) + 2 * (x1 * P * x2).sum(dim=1)
        return S
    
    def forward(self, x1, x2):
        x1 = self.extract_plda_embeddings(x1)
        x2 = self.extract_plda_embeddings(x2)
        S = self.forward_from_plda_embeddings(x1,x2)
        return S
    
    def softcdet(self, output, target):
        sigmoid = nn.Sigmoid()
        losses = [((sigmoid(self.alpha * (self.threshold[beta] - output)) * target).sum() / (target.sum()) + beta * (sigmoid(self.alpha * (output - self.threshold[beta])) * (1 - target)).sum() / ((1 - target).sum())) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def crossentropy(self, output, target):
        sigmoid = nn.Sigmoid()
        loss = F.binary_cross_entropy(sigmoid(output - self.threshold_Xent), target)
        return loss
    
    def loss(self, output, target):
        if self.lossfn == 'SoftCdet':
            return self.softcdet(output, target)
        elif self.lossfn == 'crossentropy':
            return self.crossentropy(output, target)

    def cdet(self, output, target):
        losses = [((output < self.threshold[beta]).float() * target).sum() / (target.sum()) + beta * ((output > self.threshold[beta]).float() * (1 - target)).sum() / ((1 - target).sum()) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def minc(self, output, target, update_thresholds=False, showplots=False):
        scores_target, _ = torch.sort(output[target>0.5])
        scores_nontarget, _ = torch.sort(-output[target<0.5])
        scores_nontarget = -scores_nontarget
        pmiss_arr = [arr2val(torch.where(scores_target < i)[0], -1) for i in scores_target]
        pmiss = torch.tensor(pmiss_arr).float() / (target.cpu().sum())
        pfa_arr = [arr2val(torch.where(scores_nontarget >= i)[0], -1) for i in scores_target]
        pfa = torch.tensor(pfa_arr).float() / ((1-target.cpu()).sum())
        cdet_arr, minc_dict, minc_threshold = {}, {}, {}
        for beta in self.beta:
            cdet_arr[beta] = pmiss + beta*pfa
            minc_dict[beta], thidx = torch.min(cdet_arr[beta], 0)
            minc_threshold[beta] = scores_target[thidx]
            if update_thresholds:
                self.state_dict()["Th{}".format(int(beta))].data.copy_(minc_threshold[beta])
        mincs = list(minc_dict.values())
        minc_avg = sum(mincs)/len(mincs)
        if showplots:
            plt.figure()
            minsc = output.min()
            maxsc = output.max()
            plt.hist(np.asarray(scores_nontarget), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.hist(np.asarray(scores_target), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.plot(scores_target, pmiss)
            plt.plot(scores_target, pfa)
            plt.plot(scores_target, cdet_arr[99])
            plt.plot(scores_target, cdet_arr[199])
            # plt.ylim([0,3])
            # plt.xlim([0,1.4])
            plt.show()
        return minc_avg, minc_threshold

    def LoadParamsFromKaldi(self, xvec_etdnn_pickle_file, mean_vec_file, transform_mat_file, PldaFile):
        self.xvector_extractor.LoadFromKaldi(xvec_etdnn_pickle_file)
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
        self.alpha = torch.tensor(nc.alpha).to(nc.device)
        self.beta = nc.beta
        self.dropout = nn.Dropout(p=0.5)
        self.lossfn = nc.loss


    def extract_plda_embeddings(self, x):
        x = self.centering_and_LDA(x)
        x = F.normalize(x)
        x = self.centering_and_wccn_plda(x)
        return x

    def forward_from_plda_embeddings(self,x1,x2):
        P = self.P_sqrt * self.P_sqrt
        Q = self.Q
        S = (x1 * Q * x1).sum(dim=1) + (x2 * Q * x2).sum(dim=1) + 2 * (x1 * P * x2).sum(dim=1)
        return S
    
    def forward(self, x1, x2):
        x1 = self.extract_plda_embeddings(x1)
        x2 = self.extract_plda_embeddings(x2)
        S = self.forward_from_plda_embeddings(x1,x2)
        return S

    def softcdet(self, output, target):
        sigmoid = nn.Sigmoid()
        losses = [((sigmoid(self.alpha * (self.threshold[beta] - output)) * target).sum() / (target.sum()) + beta * (sigmoid(self.alpha * (output - self.threshold[beta])) * (1 - target)).sum() / ((1 - target).sum())) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def crossentropy(self, output, target):
        sigmoid = nn.Sigmoid()
        loss = F.binary_cross_entropy(sigmoid(output - self.threshold_Xent), target)
        return loss
    
    def loss(self, output, target):
        if self.lossfn == 'SoftCdet':
            return self.softcdet(output, target)
        elif self.lossfn == 'crossentropy':
            return self.crossentropy(output, target)

    def cdet(self, output, target):
        losses = [((output < self.threshold[beta]).float() * target).sum() / (target.sum()) + beta * ((output > self.threshold[beta]).float() * (1 - target)).sum() / ((1 - target).sum()) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def minc(self, output, target, update_thresholds=False, showplots=False):
        scores_target, _ = torch.sort(output[target>0.5])
        scores_nontarget, _ = torch.sort(-output[target<0.5])
        scores_nontarget = -scores_nontarget
        pmiss_arr = [arr2val(torch.where(scores_target < i)[0], -1) for i in scores_target]
        pmiss = torch.tensor(pmiss_arr).float() / (target.cpu().sum())
        pfa_arr = [arr2val(torch.where(scores_nontarget >= i)[0], -1) for i in scores_target]
        pfa = torch.tensor(pfa_arr).float() / ((1-target.cpu()).sum())
        cdet_arr, minc_dict, minc_threshold = {}, {}, {}
        for beta in self.beta:
            cdet_arr[beta] = pmiss + beta*pfa
            minc_dict[beta], thidx = torch.min(cdet_arr[beta], 0)
            minc_threshold[beta] = scores_target[thidx]
            if update_thresholds:
                self.state_dict()["Th{}".format(int(beta))].data.copy_(minc_threshold[beta])
        mincs = list(minc_dict.values())
        minc_avg = sum(mincs)/len(mincs)
        if showplots:
            plt.figure()
            minsc = output.min()
            maxsc = output.max()
            plt.hist(np.asarray(scores_nontarget), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.hist(np.asarray(scores_target), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.plot(scores_target, pmiss)
            plt.plot(scores_target, pfa)
            plt.plot(scores_target, cdet_arr[99])
            plt.plot(scores_target, cdet_arr[199])
            # plt.ylim([0,3])
            # plt.xlim([0,1.4])
            plt.show()
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

class DPlda(nn.Module):
    def __init__(self, nc):
        super(DPlda, self).__init__()
        self.centering_and_LDA = nn.Linear(nc.xvector_dim, nc.layer1_LDA_dim)  # Centering, wccn
        self.logistic_regres = nn.Linear(nc.layer1_LDA_dim*nc.layer1_LDA_dim*2+nc.layer1_LDA_dim,1)
        self.threshold = {}
        for beta in nc.beta:
            self.threshold[beta] = nn.Parameter(0*torch.rand(1, requires_grad=True))
            self.register_parameter("Th{}".format(int(beta)), self.threshold[beta])
        self.alpha = torch.tensor(nc.alpha).to(nc.device)
        self.beta = nc.beta
        self.dropout = nn.Dropout(p=0.5)
        self.lossfn = nc.loss


    def extract_plda_embeddings(self, x):
        x = self.centering_and_LDA(x)
        x = F.normalize(x)
        return x

    def forward_from_plda_embeddings(self,x1,x2):
        x_between = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)).reshape(x1.shape[0],-1) + torch.bmm(x2.unsqueeze(2), x1.unsqueeze(1)).reshape(x1.shape[0],-1)
        x_within = torch.bmm(x1.unsqueeze(2), x1.unsqueeze(1)).reshape(x1.shape[0],-1) + torch.bmm(x2.unsqueeze(2), x2.unsqueeze(1)).reshape(x1.shape[0],-1)
        x_sum = x1+x2
        x = torch.cat((x_between,x_within,x_sum),dim=1)
        S = self.logistic_regres(x)[:,0]
        return S
    
    def forward(self, x1, x2):
        x1 = self.extract_plda_embeddings(x1)
        x2 = self.extract_plda_embeddings(x2)
        S = self.forward_from_plda_embeddings(x1,x2)
        return S

    def softcdet(self, output, target):
        sigmoid = nn.Sigmoid()
        losses = [((sigmoid(self.alpha * (self.threshold[beta] - output)) * target).sum() / (target.sum()) + beta * (sigmoid(self.alpha * (output - self.threshold[beta])) * (1 - target)).sum() / ((1 - target).sum())) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def crossentropy(self, output, target):
        sigmoid = nn.Sigmoid()
        loss = F.binary_cross_entropy(sigmoid(output), target)
        return loss
    
    def loss(self, output, target):
        if self.lossfn == 'SoftCdet':
            return self.softcdet(output, target)
        elif self.lossfn == 'crossentropy':
            return self.crossentropy(output, target)

    def cdet(self, output, target):
        losses = [((output < self.threshold[beta]).float() * target).sum() / (target.sum()) + beta * ((output > self.threshold[beta]).float() * (1 - target)).sum() / ((1 - target).sum()) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def minc(self, output, target, update_thresholds=False, showplots=False):
        scores_target, _ = torch.sort(output[target>0.5])
        scores_nontarget, _ = torch.sort(-output[target<0.5])
        scores_nontarget = -scores_nontarget
        pmiss_arr = [arr2val(torch.where(scores_target < i)[0], -1) for i in scores_target]
        pmiss = torch.tensor(pmiss_arr).float() / (target.cpu().sum())
        pfa_arr = [arr2val(torch.where(scores_nontarget >= i)[0], -1) for i in scores_target]
        pfa = torch.tensor(pfa_arr).float() / ((1-target.cpu()).sum())
        cdet_arr, minc_dict, minc_threshold = {}, {}, {}
        for beta in self.beta:
            cdet_arr[beta] = pmiss + beta*pfa
            minc_dict[beta], thidx = torch.min(cdet_arr[beta], 0)
            minc_threshold[beta] = scores_target[thidx]
            if update_thresholds:
                self.state_dict()["Th{}".format(int(beta))].data.copy_(minc_threshold[beta])
        mincs = list(minc_dict.values())
        minc_avg = sum(mincs)/len(mincs)
        if showplots:
            plt.figure()
            minsc = output.min()
            maxsc = output.max()
            plt.hist(np.asarray(scores_nontarget), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.hist(np.asarray(scores_target), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.plot(scores_target, pmiss)
            plt.plot(scores_target, pfa)
            plt.plot(scores_target, cdet_arr[99])
            plt.plot(scores_target, cdet_arr[199])
            # plt.ylim([0,3])
            # plt.xlim([0,1.4])
            plt.show()
        return minc_avg, minc_threshold
            
        
        

    def LoadParamsFromKaldi(self, mean_vec_file, transform_mat_file):
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
    
    def softcdet(self, output, target):
        sigmoid = nn.Sigmoid()
        losses = [((sigmoid(self.alpha * (self.threshold[beta] - output)) * target).sum() / (target.sum()) + beta * (sigmoid(self.alpha * (output - self.threshold[beta])) * (1 - target)).sum() / ((1 - target).sum())) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def crossentropy(self, output, target):
        sigmoid = nn.Sigmoid()
        loss = F.binary_cross_entropy(sigmoid(output), target)
        return loss
    
    def loss(self, output, target):
        if self.lossfn == 'SoftCdet':
            return self.softcdet(output, target)
        elif self.lossfn == 'crossentropy':
            return self.crossentropy(output, target)

    def cdet(self, output, target):
        losses = [((output < self.threshold[beta]).float() * target).sum() / (target.sum()) + beta * ((output > self.threshold[beta]).float() * (1 - target)).sum() / ((1 - target).sum()) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def minc(self, output, target, update_thresholds=False, showplots=False):
        scores_target, _ = torch.sort(output[target>0.5])
        scores_nontarget, _ = torch.sort(-output[target<0.5])
        scores_nontarget = -scores_nontarget
        pmiss_arr = [arr2val(torch.where(scores_target < i)[0], -1) for i in scores_target]
        pmiss = torch.tensor(pmiss_arr).float() / (target.cpu().sum())
        pfa_arr = [arr2val(torch.where(scores_nontarget >= i)[0], -1) for i in scores_target]
        pfa = torch.tensor(pfa_arr).float() / ((1-target.cpu()).sum())
        cdet_arr, minc_dict, minc_threshold = {}, {}, {}
        for beta in self.beta:
            cdet_arr[beta] = pmiss + beta*pfa
            minc_dict[beta], thidx = torch.min(cdet_arr[beta], 0)
            minc_threshold[beta] = scores_target[thidx]
            if update_thresholds:
                self.state_dict()["Th{}".format(int(beta))].data.copy_(minc_threshold[beta])
        mincs = list(minc_dict.values())
        minc_avg = sum(mincs)/len(mincs)
        if showplots:
            plt.figure()
            minsc = output.min()
            maxsc = output.max()
            plt.hist(np.asarray(scores_nontarget), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.hist(np.asarray(scores_target), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.plot(scores_target, pmiss)
            plt.plot(scores_target, pfa)
            plt.plot(scores_target, cdet_arr[99])
            plt.plot(scores_target, cdet_arr[199])
            # plt.ylim([0,3])
            # plt.xlim([0,1.4])
            plt.show()
        return minc_avg, minc_threshold

    def LoadPldaParamsFromKaldi(self, mean_vec_file, transform_mat_file):
        transform_mat = np.asarray([w.split() for w in np.asarray(subprocess.check_output(["copy-matrix","--binary=false", transform_mat_file, "-"]).decode('utf-8').strip()[2:-2].split('\n'))]).astype(float)
        mean_vec = np.asarray(subprocess.check_output(["copy-vector", "--binary=false", mean_vec_file, "-"]).decode('utf-8').strip()[1:-2].split()).astype(float)
        mdsd = self.state_dict()
        mdsd['centering_and_LDA.weight'].data.copy_(torch.from_numpy(transform_mat[:,:-1]).float())
        mdsd['centering_and_LDA.bias'].data.copy_(torch.from_numpy(transform_mat[:,-1]-transform_mat[:,:-1].dot(mean_vec)).float())
        
    def SaveModel(self, filename):
        with open(filename,'wb') as f:
            pickle.dump(self,f)