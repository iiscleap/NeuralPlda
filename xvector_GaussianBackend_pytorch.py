#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:15:54 2019

@author: shreyasr, prashantk
"""


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
import numpy as np
import pickle
import sys
from sv_trials_loaders import get_train_valid_loader, dataloader_from_trial, get_spk2xvector, dataloader_from_trials_list, dataloader_from_sre08_10_trial, generate_scores_from_net, xv_pairs_from_trial, dataset_from_list, concatenate_datasets, get_train_dataset, dataset_from_trial, dataset_from_sre08_10_trial
if '/home/data1/SRE18/shreyasr/nist_slre_toolkit' not in sys.path:
    sys.path.append('/home/data1/SRE18/shreyasr/nist_slre_toolkit')
from lib.calibration_cmn2 import get_minC_threshold, get_cmn2_thresholds
from pdb import set_trace as bp
if '/home/data2/shreyasr/nnet3_compute_experiments' not in sys.path:
    sys.path.append('/home/data2/shreyasr/nnet3_compute_experiments')
from kaldiPlda2numpydict import kaldiPlda2numpydict
import subprocess
from sre08_10_prep import get_sre08_trials_etc, get_sre10_trials_etc
from datetime import datetime
import logging

timestamp = int(datetime.timestamp(datetime.now()))
logging.basicConfig(filename='logs/kaldiplda_{}.log'.format(timestamp),
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)


class GaussianBackend(nn.Module): #Lukas 2011 Paper with Kaldi's centering, LDA and length norm.
    def __init__(self, xdim=512, LDA_dim=170, device=torch.device("cpu")):
        super(GaussianBackend, self).__init__()
        self.centering_and_LDA = nn.Linear(xdim, LDA_dim) #Centering, wccn
        self.centering_and_LDA.weight.requires_grad = False
        self.centering_and_LDA.bias.requires_grad = False
        self.alpha = torch.tensor(5.0).to(device)
        self.paired_mean_target = torch.rand(2*LDA_dim)
        self.paired_cov_inv_target = torch.rand(2*LDA_dim,2*LDA_dim)
        self.paired_mean_nontarget = torch.rand(2*LDA_dim)
        self.paired_cov_inv_nontarget = torch.rand(2*LDA_dim,2*LDA_dim)
        self.threshold1 = torch.tensor(0.0).to(device)
        self.threshold2 = torch.tensor(0.0).to(device)
        self.threshold_Xent = torch.tensor(0.0).to(device)
        # self.mu = torch.zeros(xdim) #torch.from_numpy() #Replicating centering
        # self.stddev = torch.ones(xdim) #Replicating whitening
        # self.LDA = nn.Linear(xdim, LDA_dim, bias=False) #LDA
#        self.logistic_regres = nn.Linear(LDA_dim*LDA_dim*2+LDA_dim,1)
        
		
    def forward(self, x1, x2):
#        bp()
        x1 = self.centering_and_LDA(x1) #(x1-self.mu)/self.stddev
        x2 = self.centering_and_LDA(x2) #(x2-self.mu)/self.stddev
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
#        x_between = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)).reshape(x1.shape[0],-1) + torch.bmm(x2.unsqueeze(2), x1.unsqueeze(1)).reshape(x1.shape[0],-1)
#        x_within = torch.bmm(x1.unsqueeze(2), x1.unsqueeze(1)).reshape(x1.shape[0],-1) + torch.bmm(x2.unsqueeze(2), x2.unsqueeze(1)).reshape(x1.shape[0],-1)
#        x_sum = x1+x2
        x = torch.cat((x1,x2),dim=1)
        St = (-((x-self.paired_mean_target).mm(self.paired_cov_inv_target))*(x-self.paired_mean_target)).sum(dim=1) #self.logistic_regres(x)[:,0]
        Snt = (-((x-self.paired_mean_nontarget).mm(self.paired_cov_inv_nontarget))*(x-self.paired_mean_nontarget)).sum(dim=1)
        S = St - Snt
#        x1 = self.centering_and_wccn_plda(x1)
#        x2 = self.centering_and_wccn_plda(x2)
#        P = self.P_sqrt*self.P_sqrt
#        Q = self.Q #(self.Q + self.Q.t())/2 #self.Q_sqrt.mm(self.Q_sqrt.t())
#        bp()
#        S = (x1*Q*x1).sum(dim=1) + (x2*Q*x2).sum(dim=1) + 2*(x1*P*x2).sum(dim=1)
#        x = F.relu(self.conv1(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = x.view(-1, 4*4*50)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
        return S
    
    def forward_getpaired(self, x1, x2):
#        bp()
        x1 = self.centering_and_LDA(x1) #(x1-self.mu)/self.stddev
        x2 = self.centering_and_LDA(x2) #(x2-self.mu)/self.stddev
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
#        x_between = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)).reshape(x1.shape[0],-1) + torch.bmm(x2.unsqueeze(2), x1.unsqueeze(1)).reshape(x1.shape[0],-1)
#        x_within = torch.bmm(x1.unsqueeze(2), x1.unsqueeze(1)).reshape(x1.shape[0],-1) + torch.bmm(x2.unsqueeze(2), x2.unsqueeze(1)).reshape(x1.shape[0],-1)
#        x_sum = x1+x2
        x = torch.cat((x1,x2),dim=1)
#        St = (-((x-self.paired_mean_target).mm(self.paired_cov_inv_target))*(x-self.paired_mean_target)).sum(dim=1) #self.logistic_regres(x)[:,0]
#        Snt = (-((x-self.paired_mean_nontarget).mm(self.paired_cov_inv_nontarget))*(x-self.paired_mean_nontarget)).sum(dim=1)
#        S = St - Snt
#        x1 = self.centering_and_wccn_plda(x1)
#        x2 = self.centering_and_wccn_plda(x2)
#        P = self.P_sqrt*self.P_sqrt
#        Q = self.Q #(self.Q + self.Q.t())/2 #self.Q_sqrt.mm(self.Q_sqrt.t())
#        bp()
#        S = (x1*Q*x1).sum(dim=1) + (x2*Q*x2).sum(dim=1) + 2*(x1*P*x2).sum(dim=1)
#        x = F.relu(self.conv1(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = x.view(-1, 4*4*50)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
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
            
    def to_device(self, device):
        self = self.to(device)
        self.threshold1 = self.threshold1.to(device)
        self.threshold2 = self.threshold2.to(device)
        self.threshold_Xent = self.threshold_Xent.to(device)
        self.alpha = self.alpha.to(device)
        
# LoadPldaParamsFromKaldi('../../prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre18_combined/mean.vec', '../../prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre18_combined/transform.mat','../../prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre18_combined/plda')


    
def train(args, model, device, train_loader, optimizer, epoch):    
    model.eval()
    target_sum = torch.zeros(model.paired_mean_target.shape)
    non_target_sum = torch.zeros(model.paired_mean_target.shape)
    target_sq_sum = torch.zeros(model.paired_cov_inv_target.shape)
    non_target_sq_sum = torch.zeros(model.paired_cov_inv_target.shape)
    target_count = 0
    non_target_count = 0
    with torch.no_grad():
        for data1, data2, target in train_loader:
#            bp()
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
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
    
    


def score_18_eval(sre18_eval_trials_file_path, model, device, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2):
    generate_scores_from_net("scores/scores_kaldipldanet_CUDA_GB_eval_{}.txt".format(timestamp), device, sre18_eval_trials_file_path, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2, model)


def main_score_eval():
    device = torch.device('cuda')
    model = pickle.load(open('/home/data2/SRE2019/shreyasr/X/models/kaldi_pldaNet_sre0410_swbd_16_1.swbdsremx6epoch.1571827115.pt','rb'))
    model = model.to(device)
    sre18_eval_trials_file_path = "/home/data/SRE2019/LDC2019E59/eval/docs/sre18_eval_trials.tsv"
    trial_file_path = "/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_test/trials"
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_test/xvectors.pkl"
    enroll_spk2xvectors = get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path)
    test_xvectors = pickle.load(open(test_xvector_path,'rb'))
#    sre18_eval_trials_loader = dataloader_from_trial(trial_file_paths_list[1], enroll_spk2xvectors_list[1], test_xvectors_list[1],batch_size = 2048, shuffle=True)
    sre18_eval_xv_pairs_1,sre18_eval_xv_pairs_2 = xv_pairs_from_trial(trial_file_path, enroll_spk2xvectors, test_xvectors)
    score_18_eval(sre18_eval_trials_file_path, model, device, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2)



def validate(args, model, device, data_loader, epoch, savescores=False):
#    bp()
    model.eval()
#    minC_threshold = compute_minc_threshold(args, model, device, data_loader,epoch,savescores)
    minC_threshold1, minC_threshold2, min_cent_threshold = compute_minc_threshold(args, model, device, data_loader,epoch,savescores)
    test_loss = 0
    correct = 0
    fa1 = 0
    miss1 = 0
    fa2 = 0
    miss2 = 0    
    tgt_count = 0
    non_tgt_count = 0
    #bp()
    with torch.no_grad():
        for data1, data2, target in data_loader:
#            bp()
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1,data2)
            sigmoid = nn.Sigmoid()
            test_loss += F.binary_cross_entropy(sigmoid(output-min_cent_threshold), target).item() # sum up batch loss
            correct_preds = (((output-min_cent_threshold)>0).float()==target).float() #output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
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


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def compute_minc_threshold(args, model, device, data_loader, epoch, save_scores_targets = False):
    #bp()
    device1=torch.device('cpu')
    model.to_device(device1)
    
    with torch.no_grad():
        targets, scores = np.asarray([]), np.asarray([])
        for data1, data2, target in data_loader:
            data1, data2, target = data1.to(device1), data2.to(device1), target.to(device1)
            targets = np.concatenate((targets, np.asarray(target)))
            scores = np.concatenate((scores, np.asarray(model.forward(data1,data2))))
    if save_scores_targets:
        np.save('scoresntargets/scores_{}.npy'.format(epoch),scores)
        np.save('scoresntargets/targets_{}.npy'.format(epoch),targets)
#        scores_nontarget = -np.sort(-scores[targets<0.5])
#        scores_target = np.sort(scores[targets>0.5])
#        scores_sorted = np.sort(scores)
#        pmiss = np.asarray([arr2val(np.where(scores_target<i)[0],-1) for i in scores_sorted])/len(scores_target)
#        pfa = np.asarray([arr2val(np.where(scores_nontarget>=i)[0],-1) for i in scores_sorted])/len(scores_nontarget)
#        minc_idx = np.argmin(pmiss+149*pfa)
    #bp()
#    minc_threshold = get_minC_threshold(scores,targets) #scores_sorted[minc_idx]
    minC_threshold1, minC_threshold2, min_cent_threshold = get_cmn2_thresholds(scores,targets)
    model.to_device(device)
    return minC_threshold1, minC_threshold2, min_cent_threshold

def arr2val(x,retidx):
    if x.size>0:
        return x[retidx]
    else:
        return 0

def soft_Cdet(output,target):
    softCdet = ((1-output)*target).sum()/target.sum()+149*(output*(1-target)).sum()/(1-target).sum()
    return softCdet

def main_GB():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
#    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
#    random.setstate()
    

    
    logging.info("Started at {}\n\n GaussianBackend with Len Norm. Batch size = 2048. Threshold not updated after epoch \n\n ".format(datetime.now()))

#    device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device('cpu')
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
#    bp()
    train_set = get_train_dataset(data_dir_list, xvector_scp_list, batch_size=2048)
    datasets.append(train_set)
    
    trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_dev_test/trials")
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_dev_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_test/xvectors.pkl"
    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
    sre18_dev_trials_loader = dataloader_from_trial(trial_file_paths_list[0], enroll_spk2xvectors_list[0], test_xvectors_list[0], batch_size = 2048, shuffle=True)
#    sre18_dev_trials_dataset = dataset_from_trial(trial_file_paths_list[0], enroll_spk2xvectors_list[0], test_xvectors_list[0], batch_size = 2048, shuffle=True)
#    datasets.append(sre18_dev_trials_dataset)
    sre18_dev_xv_pairs_1,sre18_dev_xv_pairs_2 = xv_pairs_from_trial(trial_file_paths_list[0], enroll_spk2xvectors_list[0], test_xvectors_list[0])
    
    #trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_test/trials")
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_test/xvectors.pkl"
    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
    #sre18_eval_trials_loader = dataloader_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors,batch_size = 256, shuffle=True)
    
#    trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_test/trials")
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre16_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre16_eval_test/xvectors.pkl"
    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
#    sre16_eval_trials_loader = dataloader_from_trial(trial_file_paths_list[2], enroll_spk2xvectors_list[2], test_xvectors_list[2], batch_size = 4096, shuffle=True)
    sre16_eval_trials_dataset = dataset_from_trial(trial_file_paths_list[2], enroll_spk2xvectors_list[2], test_xvectors_list[2], batch_size = 2048, shuffle=True)
    datasets.append(sre16_eval_trials_dataset)
#    try:
#        with open('pickled_loaders/sre08_loader.pkl','rb') as f:
#            sre08_loader = pickle.load(f)
#    except:
#        trials_08, enroll_xvectors_08, enroll_model2xvector_08, all_utts_dict_08 = get_sre08_trials_etc()
#        sre08_loader = dataloader_from_sre08_10_trial(trials_08, enroll_model2xvector_08, enroll_xvectors_08, all_utts_dict_08, batch_size = 4096, shuffle=True)
#        with open('pickled_loaders/sre08_loader.pkl','wb') as f:
#            pickle.dump(sre08_loader, f)
#            
    trials_08, enroll_xvectors_08, enroll_model2xvector_08, all_utts_dict_08 = get_sre08_trials_etc()
    sre08_dataset = dataset_from_sre08_10_trial(trials_08, enroll_model2xvector_08, enroll_xvectors_08, all_utts_dict_08, batch_size = 2048, shuffle=True)
    datasets.append(sre08_dataset)
    
#    try:
#        with open('pickled_loaders/sre10_loader.pkl','rb') as f:
#            sre10_loader = pickle.load(f)
#    except:
#        trials_10, enroll_xvectors_10, enroll_model2xvector_10, all_utts_dict_10 = get_sre10_trials_etc()
#        sre10_loader = dataloader_from_sre08_10_trial(trials_10, enroll_model2xvector_10, enroll_xvectors_10, all_utts_dict_10, batch_size = 8192, shuffle=True)
#        with open('pickled_loaders/sre10_loader.pkl','wb') as f:
#            pickle.dump(sre10_loader, f, protocol=4)
#    
    trials_10, enroll_xvectors_10, enroll_model2xvector_10, all_utts_dict_10 = get_sre10_trials_etc()
    sre10_dataset = dataset_from_sre08_10_trial(trials_10, enroll_model2xvector_10, enroll_xvectors_10, all_utts_dict_10, batch_size = 2048, shuffle=True)
    datasets.append(sre10_dataset)
    
    train_loader = concatenate_datasets(datasets,batch_size=2048)

    model = GaussianBackend(device=device).to(device)
#    model = pickle.load(open('/home/data2/SRE2019/shreyasr/X/models/kaldi_pldaNet_sre0410_swbd_16_16.swbdsremx6epoch.1571651491.pt','rb'))
#    model.threshold.cpu()
    
    model.LoadPldaParamsFromKaldi('../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/mean.vec', '../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/transform.mat') #,'../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/plda')
    
#    model_kaldi = KaldiPldaNet().to(device)
    
#    model_kaldi.LoadPldaParamsFromKaldi('../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre/mean.vec', '../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre/transform.mat','../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_unlabeled/plda_adapt')
    sre18_dev_trials_file_path = "/home/data/SRE2019/LDC2019E59/dev/docs/sre18_dev_trials.tsv"
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_losses = []
    
    bestloss = 1000
    
    print("SRE18_Dev Trials:")
    logging.info("SRE16_18_dev_eval Trials:")
    valloss, minC_threshold1, minC_threshold2, min_cent_threshold  = validate(args, model, device, sre18_dev_trials_loader, 0)
    all_losses.append(valloss)
#    bp()
#    xyz = model.threshold + torch.tensor(minc_thres).to(device)
#    model.threshold.data.copy_(xyz)
#    model.threshold += torch.tensor(minc_thres).to(device) 
#    model.state_dict()['threshold'].data.copy_(torch.tensor(minC_threshold1))

#    
#    model = pickle.load(open("models/best_pldaNet_moredata_4.pt",'rb'))
    
    epoch=1
    model = train(args, model, device, train_loader, optimizer, epoch)
    print("SRE16_18_dev_eval Trials:")
    logging.info("SRE16_18_dev_eval Trials:")
    valloss, minC_threshold1, minC_threshold2, min_cent_threshold  = validate(args, model, device,sre18_dev_trials_loader, epoch)
    all_losses.append(valloss)
#        model.threshold += torch.tensor(minc_thres).to(device) 
#        model.state_dict()['threshold'].data.copy_(torch.tensor(minc_thres))
    model.SaveModel("models/kaldi_pldaNet_sre0410_swbd_16_{}.swbdsremx6epoch.{}.pt".format(epoch,timestamp))
    print("Generating scores for Epoch ",epoch)       
    generate_scores_from_net("scores/scores_kaldipldanet_CUDA_Random{}_{}.txt".format(epoch,timestamp), device, sre18_dev_trials_file_path, sre18_dev_xv_pairs_1, sre18_dev_xv_pairs_2, model)
    try:
        if all_losses[-1] < bestloss:
            bestloss = all_losses[-1]
        if (all_losses[-1] > all_losses[-2]) and (all_losses[-2] > all_losses[-3]):
            lr = lr/2
            print("REDUCING LEARNING RATE to {} since loss trend looks like {}".format(lr, all_losses[-3:]))
            logging.info("REDUCING LEARNING RATE to {} since loss trend looks like {}".format(lr, all_losses[-3:]))
            optimizer = optim.Adam(model.parameters(), lr=lr)
    except:
        pass


def finetune(pldaNetfilepath):
    print("Finetune")
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
#    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

#    device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device('cpu')
    
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
#    bp()
    train_set = get_train_dataset(data_dir_list, xvector_scp_list, batch_size=2048)
    datasets.append(train_set)
    
    trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_dev_test/trials")
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_dev_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_dev_test/xvectors.pkl"
    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
#    sre18_dev_trials_loader = dataloader_from_trial(trial_file_paths_list[0], enroll_spk2xvectors_list[0], test_xvectors_list[0], batch_size = 2048, shuffle=True)
    sre18_dev_trials_dataset = dataset_from_trial(trial_file_paths_list[0], enroll_spk2xvectors_list[0], test_xvectors_list[0], batch_size = 2048, shuffle=True)
    datasets.append(sre18_dev_trials_dataset)
    sre18_dev_xv_pairs_1,sre18_dev_xv_pairs_2 = xv_pairs_from_trial(trial_file_paths_list[0], enroll_spk2xvectors_list[0], test_xvectors_list[0])
    bp()
#    trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_test/trials")
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre16_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre16_eval_test/xvectors.pkl"
    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
#    sre16_eval_trials_loader = dataloader_from_trial(trial_file_paths_list[2], enroll_spk2xvectors_list[2], test_xvectors_list[2], batch_size = 4096, shuffle=True)
    sre16_eval_trials_dataset = dataset_from_trial(trial_file_paths_list[1], enroll_spk2xvectors_list[1], test_xvectors_list[1], batch_size = 2048, shuffle=True)
    datasets.append(sre16_eval_trials_dataset)
#    try:
#        with open('pickled_loaders/sre08_loader.pkl','rb') as f:
#            sre08_loader = pickle.load(f)
#    except:
#        trials_08, enroll_xvectors_08, enroll_model2xvector_08, all_utts_dict_08 = get_sre08_trials_etc()
#        sre08_loader = dataloader_from_sre08_10_trial(trials_08, enroll_model2xvector_08, enroll_xvectors_08, all_utts_dict_08, batch_size = 4096, shuffle=True)
#        with open('pickled_loaders/sre08_loader.pkl','wb') as f:
#            pickle.dump(sre08_loader, f)
#            
    trials_08, enroll_xvectors_08, enroll_model2xvector_08, all_utts_dict_08 = get_sre08_trials_etc()
    sre08_dataset = dataset_from_sre08_10_trial(trials_08, enroll_model2xvector_08, enroll_xvectors_08, all_utts_dict_08, batch_size = 2048, shuffle=True)
    datasets.append(sre08_dataset)
    
#    try:
#        with open('pickled_loaders/sre10_loader.pkl','rb') as f:
#            sre10_loader = pickle.load(f)
#    except:
#        trials_10, enroll_xvectors_10, enroll_model2xvector_10, all_utts_dict_10 = get_sre10_trials_etc()
#        sre10_loader = dataloader_from_sre08_10_trial(trials_10, enroll_model2xvector_10, enroll_xvectors_10, all_utts_dict_10, batch_size = 8192, shuffle=True)
#        with open('pickled_loaders/sre10_loader.pkl','wb') as f:
#            pickle.dump(sre10_loader, f, protocol=4)
#    
    trials_10, enroll_xvectors_10, enroll_model2xvector_10, all_utts_dict_10 = get_sre10_trials_etc()
    sre10_dataset = dataset_from_sre08_10_trial(trials_10, enroll_model2xvector_10, enroll_xvectors_10, all_utts_dict_10, batch_size = 2048, shuffle=True)
    datasets.append(sre10_dataset)
    
    train_loader = concatenate_datasets(datasets,batch_size=2048)
    
    sre18_eval_trials_file_path = "/home/data/SRE2019/LDC2019E59/eval/docs/sre18_eval_trials.tsv"
    trial_file_path = "/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_test/trials"
    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_enrollment/spk2utt"
    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_enrollment/xvectors.pkl"
    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre18_eval_test/xvectors.pkl"
    enroll_spk2xvectors = get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path)
    test_xvectors = pickle.load(open(test_xvector_path,'rb'))
#    sre18_eval_trials_loader = dataloader_from_trial(trial_file_paths_list[1], enroll_spk2xvectors_list[1], test_xvectors_list[1],batch_size = 2048, shuffle=True)
    sre18_eval_xv_pairs_1,sre18_eval_xv_pairs_2 = xv_pairs_from_trial(trial_file_path, enroll_spk2xvectors, test_xvectors)
    
    
#    #trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
#    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_test/trials")
#    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre18_eval_enrollment/spk2utt"
#    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre18_eval_enrollment/xvectors.pkl"
#    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre18_eval_test/xvectors.pkl"
#    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
#    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
#    #sre18_eval_trials_loader = dataloader_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors,batch_size = 256, shuffle=True)
#    
#    #trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list = [], [], []
#    trial_file_paths_list.append("/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_test/trials")
#    enroll_spk2utt_path="/home/data2/SRE2019/prashantk/voxceleb/v3/data/sre16_eval_enrollment/spk2utt"
#    enroll_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre16_eval_enrollment/xvectors.pkl"
#    test_xvector_path="/home/data2/SRE2019/prashantk/voxceleb/v3/exp/xvector_nnet_2a/xvectors_sre16_eval_test/xvectors.pkl"
#    enroll_spk2xvectors_list.append(get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path))
#    test_xvectors_list.append(pickle.load(open(test_xvector_path,'rb')))
    #sre16_eval_trials_loader = dataloader_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors,batch_size = 256, shuffle=True)
    
#    all_trials_loader = dataloader_from_trials_list(trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list, batch_size = 2048, shuffle=True)
    
    model = pickle.load(open(pldaNetfilepath,'rb'))
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train(args, model, device, train_loader, optimizer, 0)
    model.SaveModel("models/kaldipldaNet_GB_finetuned.pt")
    generate_scores_from_net("scores/scores_kaldipldanet_finetuned_GB_{}.txt".format(timestamp), device, sre18_eval_trials_file_path, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2, model)
#        test(args, model, device, test_loader)

#    if (args.save_model):
#        torch.save(model.state_dict(),"mnist_cnn.pt")
        
    

if __name__ == '__main__':
#    main_GB()
#    main_score_eval()
    finetune('models/kaldi_pldaNet_sre0410_swbd_16_1.swbdsremx6epoch.1571827115.pt')
