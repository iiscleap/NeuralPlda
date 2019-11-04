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
import numpy as np
import pickle
import subprocess

from utils.sv_trials_loaders import dataloader_from_trial, get_spk2xvector, generate_scores_from_net, generate_scores_in_batches, xv_pairs_from_trial, concatenate_datasets, get_train_dataset, dataset_from_trial, dataset_from_sre08_10_trial
from utils.calibration import get_cmn2_thresholds
from utils.Kaldi2NumpyUtils.kaldiPlda2numpydict import kaldiPlda2numpydict

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


            
class NeuralPlda(nn.Module):
    def __init__(self, xdim=512, LDA_dim=170, PLDA_dim=170, device=torch.device("cuda")):
        super(NeuralPlda, self).__init__()
        self.centering_and_LDA = nn.Linear(xdim, LDA_dim) #Centering, wccn
        self.centering_and_wccn_plda = nn.Linear(LDA_dim, PLDA_dim)
        self.P_sqrt = nn.Parameter(torch.rand(PLDA_dim,requires_grad=True))
        self.Q = nn.Parameter(torch.rand(PLDA_dim,requires_grad=True))
        self.threshold1 = nn.Parameter(torch.tensor(0.0)).to(device)
        self.threshold2 = nn.Parameter(torch.tensor(0.0)).to(device)
        self.threshold_Xent = nn.Parameter(torch.tensor(0.0)).to(device)
        self.alpha = torch.tensor(5.0).to(device)
		
    def forward(self, x1, x2):
        x1 = self.centering_and_LDA(x1) 
        x2 = self.centering_and_LDA(x2) 
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        
        x1 = self.centering_and_wccn_plda(x1)
        x2 = self.centering_and_wccn_plda(x2)
        P = self.P_sqrt*self.P_sqrt
        Q = self.Q 
        S = (x1*Q*x1).sum(dim=1) + (x2*Q*x2).sum(dim=1) + 2*(x1*P*x2).sum(dim=1)

        return S
    
    def LoadPldaParamsFromKaldi(self, mean_vec_file, transform_mat_file, PldaFile):
        plda = kaldiPlda2numpydict(PldaFile)
        transform_mat = np.asarray([w.split() for w in np.asarray(subprocess.check_output(["copy-matrix","--binary=false", transform_mat_file, "-"]).decode('utf-8').strip()[2:-2].split('\n'))]).astype(float)
        mean_vec = np.asarray(subprocess.check_output(["copy-vector", "--binary=false",mean_vec_file, "-"]).decode('utf-8').strip()[1:-2].split()).astype(float)
        mdsd = self.state_dict()
        mdsd['centering_and_LDA.weight'].data.copy_(torch.from_numpy(transform_mat[:,:-1]).float())
        mdsd['centering_and_LDA.bias'].data.copy_(torch.from_numpy(transform_mat[:,-1]-transform_mat[:,:-1].dot(mean_vec)).float())
        mdsd['centering_and_wccn_plda.weight'].data.copy_(torch.from_numpy(plda['diagonalizing_transform']).float())
        mdsd['centering_and_wccn_plda.bias'].data.copy_(torch.from_numpy(-plda['diagonalizing_transform'].dot(plda['plda_mean'])).float())
        mdsd['P_sqrt'].data.copy_(torch.from_numpy(np.sqrt(plda['diagP'])).float())
        mdsd['Q'].data.copy_(torch.from_numpy(plda['diagQ']).float())
        
    def SaveModel(self, filename):
        with open(filename,'wb') as f:
            pickle.dump(self,f)
        

    
def train(args, model, device, train_loader, optimizer, epoch):    
    model.train()
    softcdets = []
    crossentropies = []
    for batch_idx, (data1, data2, target) in enumerate(train_loader):    
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        sigmoid = nn.Sigmoid()
        loss1 = (sigmoid(model.alpha*(model.threshold1-output))*target).sum()/(target.sum()) + 99*(sigmoid(model.alpha*(output-model.threshold1))*(1-target)).sum()/((1-target).sum())
        loss2 = (sigmoid(model.alpha*(model.threshold2-output))*target).sum()/(target.sum()) + 199*(sigmoid(model.alpha*(output-model.threshold2))*(1-target)).sum()/((1-target).sum())
        loss_bce = F.binary_cross_entropy(sigmoid(output-model.threshold_Xent),target)
        loss = loss_bce # Change to 0.5*(loss1+loss2) + 0.1*loss_bce #or # 0.5*(loss1+loss2) #When required
        softcdets.append((loss1.item()+loss2.item())/2)
        crossentropies.append(loss_bce.item())
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t SoftCdet: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(softcdets)))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t SoftCdet: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(softcdets)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Crossentropy: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(crossentropies)))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Crossentropy: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(crossentropies)))
            softcdets = []
            crossentropies = []
            


def validate(args, model, device, data_loader):
    model.eval()
    minC_threshold1, minC_threshold2, min_cent_threshold = compute_minc_threshold(args, model, device, data_loader)
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



def compute_minc_threshold(args, model, device, data_loader):
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
    generate_scores_in_batches("scores/{}_{}.txt".format('sre18_eval',timestamp), device, sre18_eval_trials_file_path, sre18_eval_xv_pairs_1, sre18_eval_xv_pairs_2, model)

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

def main_kaldiplda():
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

    
    logging.info("Started at {}\n\n New class. GPU. 3 thresholds. Random init. Batch size = 2048. Threshold not updated after epoch \n\n ".format(datetime.now()))

    device = torch.device("cuda" if use_cuda else "cpu")
    
    
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
    
    
    model = NeuralPlda().to(device)
    ## Uncomment to initialize with a pickled pretrained model or a Kaldi PLDA model 
    
    # model = pickle.load(open('/home/data2/SRE2019/shreyasr/X/models/kaldi_pldaNet_sre0410_swbd_16_16.swbdsremx6epoch.1571651491.pt','rb'))
    
    ## To load a Kaldi trained PLDA model, Specify the paths of 'mean.vec', 'transform.mat' and 'plda' generated from stage 8 of https://github.com/kaldi-asr/kaldi/blob/master/egs/sre16/v2/run.sh 
    # model.LoadPldaParamsFromKaldi('../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/mean.vec', '../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/transform.mat','../../prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_swbd_sre_mx6/plda')
    

    sre18_dev_trials_file_path = "/home/data/SRE2019/LDC2019E59/dev/docs/sre18_dev_trials.tsv"
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_losses = []
    
    bestloss = 1000
    
    print("SRE18_Dev Trials:")
    logging.info("SRE16_18_dev_eval Trials:")
    valloss, minC_threshold1, minC_threshold2, min_cent_threshold  = validate(args, model, device, sre18_dev_trials_loader)
    all_losses.append(valloss)

    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        print("SRE16_18_dev_eval Trials:")
        logging.info("SRE16_18_dev_eval Trials:")
        valloss, minC_threshold1, minC_threshold2, min_cent_threshold  = validate(args, model, device, sre18_dev_trials_loader)
        all_losses.append(valloss)
        model.SaveModel("models/kaldi_pldaNet_sre0410_swbd_16_{}.swbdsremx6epoch.{}.pt".format(epoch,timestamp))
        print("Generating scores for Epoch ",epoch)       
        generate_scores_in_batches("scores/scores_kaldipldanet_CUDA_Random{}_{}.txt".format(epoch,timestamp), device, sre18_dev_trials_file_path, sre18_dev_xv_pairs_1, sre18_dev_xv_pairs_2, model)
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

        
if __name__ == '__main__':
    main_kaldiplda()
#    main_score_eval()
#    finetune('models/kaldi_pldaNet_sre0410_swbd_16_1.swbdsremx6epoch.1571827115.pt')
