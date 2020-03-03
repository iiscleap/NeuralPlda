#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:15:54 2019

@author: shreyasr, prashantk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import random
import pickle
import subprocess
from utils.NpldaConf import NpldaConf
from pdb import set_trace as bp
from utils.sv_trials_loaders import combine_trials_and_get_loader, get_trials_loaders_dict, load_xvec_from_numbatch, load_xvec_from_idbatch

from datetime import datetime
import logging

from utils.models import NeuralPlda

def train(nc, model, device, train_loader, mega_xvec_dict, num_to_id_dict, optimizer, epoch, valid_loaders=None):
        
    model.train()
    softcdets = []

    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        data1_xvec, data2_xvec = load_xvec_from_numbatch(mega_xvec_dict, num_to_id_dict, data1, data2, device)
        optimizer.zero_grad()
        output = model(data1_xvec, data2_xvec)
        # sigmoid = nn.Sigmoid()
        # loss_bce = F.binary_cross_entropy(sigmoid(output - model.threshold_Xent), target)
        loss = model.softcdet(output, target) # loss_bce
        softcdets.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if batch_idx % nc.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t SoftCdet: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), sum(softcdets)/len(softcdets)))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t SoftCdet: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), sum(softcdets)/len(softcdets)))
            softcdets = []



def validate(nc, model, device, mega_xvec_dict, num_to_id_dict, data_loader, update_thresholds=False):
    model.eval()
    with torch.no_grad():
        targets, scores = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for data1, data2, target in data_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            data1_xvec, data2_xvec = load_xvec_from_numbatch(mega_xvec_dict, num_to_id_dict, data1, data2,
                                                          device)  
            targets = torch.cat((targets, target))
            scores_batch = model.forward(data1_xvec, data2_xvec)
            scores = torch.cat((scores, scores_batch))
        soft_cdet_loss = model.softcdet(scores, targets)
        cdet_mdl = model.cdet(scores, targets)
        minc, minc_threshold = model.minc(scores, targets, update_thresholds=update_thresholds)
    
    logging.info('\n\nTest set: C_det (mdl): {:.4f}\n'.format(cdet_mdl))
    logging.info('Test set: soft C_det (mdl): {:.4f}\n'.format(soft_cdet_loss))
    logging.info('Test set: C_min: {:.4f}\n'.format(minc))
    for beta in nc.beta:
        logging.info('Test set: argmin threshold [{}]: {:.4f}\n'.format(beta, minc_threshold[beta]))
    
    print('\n\nTest set: C_det (mdl): {:.4f}\n'.format(cdet_mdl))
    print('Test set: soft C_det (mdl): {:.4f}\n'.format(soft_cdet_loss))
    print('Test set: C_min: {:.4f}\n'.format(minc))
    for beta in nc.beta:
        print('Test set: argmin threshold [{}]: {:.4f}\n'.format(beta, minc_threshold[beta]))
        
    return minc, minc_threshold




def main_kaldiplda():   
    timestamp = int(datetime.timestamp(datetime.now()))
    print(timestamp)
    logging.basicConfig(filename='logs/kaldiplda_{}.log'.format(timestamp),
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    # %% Configure Training
    configfile = 'conf/voices_config.cfg'
    
    nc = NpldaConf(configfile)
    
    torch.manual_seed(nc.seed)
    np.random.seed(nc.seed)
    random.seed(nc.seed)

    logging.info("Started at {}.\n".format(datetime.now()))
    

    if not torch.cuda.is_available():
        nc.device='cpu'
    device = torch.device(nc.device)
    
    print("Running on {}...".format(nc.device))
    logging.info("Running on {} ...\n".format(nc.device))
    logging.info("\nConfiguration:\n\n{}\n\n".format(''.join(open(configfile,'r').readlines())))
          
    # %%Load the generated training data trials and make loaders here

    mega_xvec_dict = pickle.load(open(nc.mega_xvector_pkl, 'rb'))
    num_to_id_dict = {i: j for i, j in enumerate(list(mega_xvec_dict))}
    id_to_num_dict = {v: k for k, v in num_to_id_dict.items()}
    
    train_loader = combine_trials_and_get_loader(nc.training_data_trials_list, id_to_num_dict, batch_size=nc.batch_size)
    
    # train_loader_sampled = combine_trials_and_get_loader(nc.training_data_trials_list, id_to_num_dict, batch_size=nc.batch_size, subset=0.05)
    
    valid_loaders_dict = get_trials_loaders_dict(nc.validation_trials_list, id_to_num_dict, batch_size=5*nc.batch_size)
    
    # %% Initialize model and stuff

    model = NeuralPlda(nc).to(device)
    
    ## To load a Kaldi trained PLDA model, Specify the paths of 'mean.vec', 'transform.mat' and 'plda' generated from stage 8 of https://github.com/kaldi-asr/kaldi/blob/master/egs/sre16/v2/run.sh 
    
    if nc.initialization == 'kaldi':
        model.LoadPldaParamsFromKaldi(nc.meanvec, nc.transformmat, nc.kaldiplda)

    ## Uncomment to initialize with a pickled pretrained model
    # model = pickle.load(open('/home/data2/SRE2019/shreyasr/X/models/kaldi_pldaNet_sre0410_swbd_16_16.swbdsremx6epoch.1571651491.pt','rb'))
        
    optimizer = optim.Adam(model.parameters(), lr=nc.lr, weight_decay=1e-5)

    print("Initializing the thresholds... Whatever numbers that get printed here are junk.\n")
    valloss, minC_threshold = validate(nc, model, device, mega_xvec_dict, num_to_id_dict, valid_loaders_dict[nc.heldout_set_for_th_init], update_thresholds=True)


    
    # %% Train and Validate model
    print("\n\nEpoch 0: After Initialization\n")
    all_losses = []
    for val_set, valid_loader in valid_loaders_dict.items():
        print("Validating {}".format(val_set))
        logging.info("Train set:")
        valloss, minC_threshold = validate(nc, model, device, mega_xvec_dict, num_to_id_dict, valid_loader)
        if val_set==nc.heldout_set_for_lr_decay:
            all_losses.append(valloss)


    for epoch in range(1, nc.n_epochs + 1):
        train(nc, model, device, train_loader , mega_xvec_dict, num_to_id_dict, optimizer, epoch)
        
        for val_set, valid_loader in valid_loaders_dict.items():
            print("Validating {}".format(val_set))
            logging.info("Train set:")
            valloss, minC_threshold = validate(nc, model, device, mega_xvec_dict, num_to_id_dict, valid_loader)
            if val_set==nc.heldout_set_for_lr_decay:
                all_losses.append(valloss)

            
        model.SaveModel("models/NPLDA.{}.pt".format(epoch, timestamp))
        for trial_file in nc.test_trials_list:
            print("Generating scores for Epoch {} with trial file {}".format(epoch, trial_file))

            nc.generate_scorefile("scores/kaldipldanet_epoch{}_{}_{}.txt".format(epoch, os.path.splitext(os.path.basename(trial_file))[0], timestamp), trial_file, mega_xvec_dict, model, device)

        try:
            if (all_losses[-1] > all_losses[-2]) and (all_losses[-2] > all_losses[-3]):
                nc.lr = nc.lr / 2
                print("REDUCING LEARNING RATE to {} since loss trend looks like {}".format(nc.lr, all_losses[-3:]))
                logging.info("REDUCING LEARNING RATE to {} since loss trend looks like {}".format(nc.lr, all_losses[-3:]))
                optimizer = optim.Adam(model.parameters(), lr=nc.lr, weight_decay=1e-5)
        except:
            pass


if __name__ == '__main__':
    main_kaldiplda()