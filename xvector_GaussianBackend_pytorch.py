#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:15:54 2019

@author: shreyasr
"""


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import subprocess

from utils.sv_trials_loaders import dataloader_from_trial, get_spk2xvector, generate_scores_from_net, xv_pairs_from_trial, concatenate_datasets, get_train_dataset, dataset_from_trial, dataset_from_sre08_10_trial
from utils.calibration import get_cmn2_thresholds

from utils.sre08_10_prep import get_sre08_trials_etc, get_sre10_trials_etc
from datetime import datetime
import logging

timestamp = int(datetime.timestamp(datetime.now()))
print(timestamp)
logging.basicConfig(filename='logs/kaldiplda_{}.log'.format(timestamp),
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)


            
class GaussianBackend(nn.Module): 
    def __init__(self, xdim=512, LDA_dim=170):
        super(GaussianBackend, self).__init__()
        self.centering_and_LDA = nn.Linear(xdim, LDA_dim)
        self.centering_and_LDA.weight.requires_grad = False
        self.centering_and_LDA.bias.requires_grad = False
        self.paired_mean_target = torch.rand(2*LDA_dim)
        self.paired_cov_inv_target = torch.rand(2*LDA_dim,2*LDA_dim)
        self.paired_mean_nontarget = torch.rand(2*LDA_dim)
        self.paired_cov_inv_nontarget = torch.rand(2*LDA_dim,2*LDA_dim)

        
		
    def forward(self, x1, x2):
        x1 = self.centering_and_LDA(x1) #(x1-self.mu)/self.stddev
        x2 = self.centering_and_LDA(x2) #(x2-self.mu)/self.stddev
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x = torch.cat((x1,x2),dim=1)
        St = (-((x-self.paired_mean_target).mm(self.paired_cov_inv_target))*(x-self.paired_mean_target)).sum(dim=1) #self.logistic_regres(x)[:,0]
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
            

    
def train(model, train_loader):    
    model.eval()
    target_sum = torch.zeros(model.paired_mean_target.shape)
    non_target_sum = torch.zeros(model.paired_mean_target.shape)
    target_sq_sum = torch.zeros(model.paired_cov_inv_target.shape)
    non_target_sq_sum = torch.zeros(model.paired_cov_inv_target.shape)
    target_count = 0
    non_target_count = 0
    with torch.no_grad():
        for data1, data2, target in train_loader:
            x = model.forward_getpaired(data1,data2)
            
            target_count += target.sum().item()
            if target.sum().item() >= 0.5:
                target_sum += x[target>0.5].sum(dim=0)
                target_sq_sum += x[target>0.5].t() @ x[target>0.5]
            
            non_target_count += (1-target).sum().item()
            if (1-target).sum().item() >= 0.5:
                non_target_sum += x[target<0.5].sum(dim=0)
                non_target_sq_sum += x[target<0.5].t() @ x[target<0.5]
    model.paired_mean_target = target_sum/target_count
    model.paired_cov_inv_target = torch.inverse(target_sq_sum/target_count - (model.paired_mean_target[:,np.newaxis] @ model.paired_mean_target[np.newaxis,:]))
    model.paired_mean_nontarget = non_target_sum/(non_target_count-1)
    model.paired_cov_inv_nontarget = torch.inverse(non_target_sq_sum/(non_target_count-1)- (model.paired_mean_nontarget[:,np.newaxis] @ model.paired_mean_nontarget[np.newaxis,:]))
    return model


def validate(model, device, data_loader):
    model.eval()
    minC_threshold1, minC_threshold2, min_cent_threshold = compute_minc_threshold(model, device, data_loader)
    test_loss = 0
    correct = 0
    fa1 = 0
    miss1 = 0
    fa2 = 0
    miss2 = 0    
    tgt_count = 0
    non_tgt_count = 0
    with torch.no_grad():
        for data1, data2, target in data_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1,data2)
            sigmoid = nn.Sigmoid()
            test_loss += F.binary_cross_entropy(sigmoid(output-min_cent_threshold), target).item()
            correct_preds = (((output-min_cent_threshold)>0).float()==target).float()
            correct += (correct_preds).sum().item()
            tgt_count += target.sum().item()
            non_tgt_count += (1-target).sum().item()
            fa1 += ((output>minC_threshold1).float()*(1-target)).sum().item()
            miss1 += ((output<minC_threshold1).float()*target).sum().item()
            fa2 += ((output>minC_threshold2).float()*(1-target)).sum().item()
            miss2 += ((output<minC_threshold2).float()*target).sum().item()
    Pmiss1 = miss1/tgt_count
    Pfa1 = fa1/non_tgt_count
    Cdet1 = Pmiss1 + 99*Pfa1
    Pmiss2 = miss2/tgt_count
    Pfa2 = fa2/non_tgt_count
    Cdet2 = Pmiss2 + 199*Pfa2
    Cdet = (Cdet1+Cdet2)/2
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    print('\nTest set: Pfa1: {:.2f}\n'.format(Pfa1))
    print('\nTest set: Pmiss1: {:.2f}\n'.format(Pmiss1))
    print('\nTest set: Pfa2: {:.2f}\n'.format(Pfa2))
    print('\nTest set: Pmiss2: {:.2f}\n'.format(Pmiss2))
    print('\nTest set: C_det(149): {:.2f}\n'.format(Cdet))
    
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    logging.info('\nTest set: Pfa1: {:.2f}\n'.format(Pfa1))
    logging.info('\nTest set: Pmiss1: {:.2f}\n'.format(Pmiss1))
    logging.info('\nTest set: Pfa2: {:.2f}\n'.format(Pfa2))
    logging.info('\nTest set: Pmiss2: {:.2f}\n'.format(Pmiss2))
    logging.info('\nTest set: C_det(149): {:.2f}\n'.format(Cdet))
    return Cdet, minC_threshold1, minC_threshold2, min_cent_threshold



def compute_minc_threshold(model, device, data_loader):
    device1=torch.device('cpu')
    model = model.to(device1)
    with torch.no_grad():
        targets, scores = np.asarray([]), np.asarray([])
        for data1, data2, target in data_loader:
            data1, data2, target = data1.to(device1), data2.to(device1), target.to(device1)
            targets = np.concatenate((targets, np.asarray(target)))
            scores = np.concatenate((scores, np.asarray(model.forward(data1,data2))))
    minC_threshold1, minC_threshold2, min_cent_threshold = get_cmn2_thresholds(scores,targets)
    model = model.to(device)
    return minC_threshold1, minC_threshold2, min_cent_threshold


def score_18_eval(sre18_eval_trials_file_path, model, device, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2):
    generate_scores_from_net("scores/scores_kaldipldanet_CUDA__random_init_xent_eval_{}.txt".format(timestamp), device, sre18_eval_trials_file_path, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2, model)

def main_score_eval():
    print("Scoring eval")
    device = torch.device('cuda')
    model = pickle.load(open('/home/data2/SRE2019/shreyasr/X/models/kaldi_pldaNet_sre0410_swbd_16_10.swbdsremx6epoch.1571810057.pt','rb'))
    model = model.to(device)
    sre18_eval_trials_file_path = "/home/data/SRE2019/LDC2019E59/eval/docs/sre18_eval_trials.tsv"
    trial_file_path = "/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_test/trials"
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_test/xvectors.pkl"
    enroll_spk2xvectors = get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path)
    test_xvectors = pickle.load(open(test_xvector_path,'rb'))
    sre18_eval_xv_pairs_1,sre18_eval_xv_pairs_2 = xv_pairs_from_trial(trial_file_path, enroll_spk2xvectors, test_xvectors)
    score_18_eval(sre18_eval_trials_file_path, model, device, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2)
    print("Done")

def main_GB():
    torch.manual_seed(1)

    
    logging.info("Started at {}\n\n New class. GPU. 3 thresholds. Random init. Batch size = 2048. Threshold not updated after epoch \n\n ".format(datetime.now()))

    device = torch.device("cpu")
    
    
    ###########################################################################
    # Generating training data loaders here
    ###########################################################################
    
    datasets=[]
    
    data_dir_list = np.asarray([['/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre2004/male','5'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre2004/female','5'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre_2005_2006_08/male','7'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre_2005_2006_08/female','7'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre10/male','10'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre10/female','10'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/swbd/male','2'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/swbd/female','2'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/mx6/grepd/male','4'],
                                ['/home/data2/SRE2019/prashantk/voxceleb/v3/data/mx6/grepd/female','5']])
    

    xvector_scp_list = np.asarray(['/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd/xvector_fullpaths.scp',
                                   '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre/xvector_fullpaths.scp',
                                   '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_mx6/xvector_fullpaths.scp'])

    train_set = get_train_dataset(data_dir_list, xvector_scp_list, batch_size=2048)
    datasets.append(train_set)
    
    
    # NOTE: 'xvectors.pkl' files are generated using utils/Kaldi2NumpyUtils/kaldivec2numpydict.py
    
    trial_file_path = "/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_dev_test/trials"
    enroll_spk2utt_path = "/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_dev_enrollment/spk2utt"
    enroll_xvector_path = "/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_enrollment/xvectors.pkl"
    test_xvector_path = "/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_test/xvectors.pkl"
    enroll_spk2xvector = get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path)
    test_xvectors = pickle.load(open(test_xvector_path,'rb'))
    sre18_dev_trials_loader = dataloader_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors, batch_size = 2048, shuffle=True)
    sre18_dev_xv_pairs_1,sre18_dev_xv_pairs_2 = xv_pairs_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors)
    

    trial_file_path = "/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_test/trials"
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre16_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre16_eval_test/xvectors.pkl"
    enroll_spk2xvector = get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path)
    test_xvectors = pickle.load(open(test_xvector_path,'rb'))
    sre16_eval_trials_dataset = dataset_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors, batch_size = 2048, shuffle=True)
    datasets.append(sre16_eval_trials_dataset)

        
    trials_08, enroll_xvectors_08, enroll_model2xvector_08, all_utts_dict_08 = get_sre08_trials_etc()
    sre08_dataset = dataset_from_sre08_10_trial(trials_08, enroll_model2xvector_08, enroll_xvectors_08, all_utts_dict_08, batch_size = 2048, shuffle=True)
    
    datasets.append(sre08_dataset)
    
    trials_10, enroll_xvectors_10, enroll_model2xvector_10, all_utts_dict_10 = get_sre10_trials_etc()
    sre10_dataset = dataset_from_sre08_10_trial(trials_10, enroll_model2xvector_10, enroll_xvectors_10, all_utts_dict_10, batch_size = 2048, shuffle=True)
    datasets.append(sre10_dataset)
    
    train_loader = concatenate_datasets(datasets,batch_size=2048)
    
    ###########################################################################
    # Fishished generating training data loaders
    ###########################################################################
    
    
    model = GaussianBackend()
    ## Uncomment to initialize with a pickled pretrained model or a Kaldi PLDA model 
    
    # model = pickle.load(open('/home/data2/SRE2019/shreyasr/X/models/GaussianBackend_swbd_sre_mx6.1571651491.pt','rb'))
    
    ## To load a Kaldi trained PLDA model, Specify the paths of 'mean.vec', 'transform.mat' and 'plda' generated from stage 8 of https://github.com/kaldi-asr/kaldi/blob/master/egs/sre16/v2/run.sh 
    # model.LoadPldaParamsFromKaldi('../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/mean.vec', '../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/transform.mat','../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/plda')
    

    sre18_dev_trials_file_path = "/home/data/SRE2019/LDC2019E59/dev/docs/sre18_dev_trials.tsv"

    print("SRE18_Dev Trials:")
    logging.info("SRE16_18_dev_eval Trials:")
    _,_,_,_ =  validate(model, device, sre18_dev_trials_loader)


    train(model, train_loader)
    print("SRE18_Dev Trials:")
    logging.info("SRE16_18_dev_eval Trials:")
    _,_,_,_ =  validate(model, device, sre18_dev_trials_loader)
    model.SaveModel("models/GaussianBackend_swbd_sre_mx6.{}.pt".format(timestamp))
    print("Generating scores for GB")       
    generate_scores_from_net("scores/scores_GaussianBackend_{}.txt".format(timestamp), device, sre18_dev_trials_file_path, sre18_dev_xv_pairs_1, sre18_dev_xv_pairs_2, model)


        
if __name__ == '__main__':
    main_GB()
#    main_score_eval()
#    finetune('models/kaldi_pldaNet_sre0410_swbd_16_1.swbdsremx6epoch.1571827115.pt')
