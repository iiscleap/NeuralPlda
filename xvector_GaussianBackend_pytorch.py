#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:15:54 2019

@author: shreyasr
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

from utils.models import GaussianBackend

    
def train(nc, model, train_loader, mega_xvec_dict, num_to_id_dict):    
    model.eval()
    target_sum = torch.zeros(model.paired_mean_target.shape)
    non_target_sum = torch.zeros(model.paired_mean_target.shape)
    target_sq_sum = torch.zeros(model.paired_cov_inv_target.shape)
    non_target_sq_sum = torch.zeros(model.paired_cov_inv_target.shape)
    target_count = 0
    non_target_count = 0
    with torch.no_grad():
        for data1, data2, target in train_loader:
            data1_xvec, data2_xvec = load_xvec_from_numbatch(mega_xvec_dict, num_to_id_dict, data1, data2, device=torch.device('cpu'))
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


def validate(nc, model, data_loader, mega_xvec_dict, num_to_id_dict, device=torch.device('cpu')):
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
        minc, minc_threshold = model.minc(scores, targets)
    
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


def main_GB():
    
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

    logging.info("Started at {}.\n\n GAUSSIAN BACKEND \n\n".format(datetime.now()))
    
    nc.device='cpu' #CPU enough for GB
    
    print("Running on {}...".format(nc.device))
    logging.info("\nConfiguration:\n\n{}\n\n".format(''.join(open(configfile,'r').readlines())))
    logging.info("Running on {} ...\n".format(nc.device))
          
    # %%Load the generated training data trials and make loaders here

    mega_xvec_dict = pickle.load(open(nc.mega_xvector_pkl, 'rb'))
    num_to_id_dict = {i: j for i, j in enumerate(list(mega_xvec_dict))}
    id_to_num_dict = {v: k for k, v in num_to_id_dict.items()}
    
    train_loader = combine_trials_and_get_loader(nc.training_data_trials_list, id_to_num_dict, batch_size=nc.batch_size)
    
    # train_loader_sampled = combine_trials_and_get_loader(nc.training_data_trials_list, id_to_num_dict, batch_size=nc.batch_size, subset=0.05)
    
    valid_loaders_dict = get_trials_loaders_dict(nc.validation_trials_list, id_to_num_dict, batch_size=5*nc.batch_size)
    
    model = GaussianBackend()

    train(nc, model, train_loader, mega_xvec_dict, num_to_id_dict)
    
    for val_set, valid_loader in valid_loaders_dict.items():
        print("Validating {}".format(val_set))
        minc, minc_threshold =  validate(nc, model, valid_loader, mega_xvec_dict, num_to_id_dict)
    
    model.SaveModel("models/GaussianBackend_swbd_sre_mx6.{}.pt".format(timestamp))
    for trial_file in nc.test_trials_list:
            print("Generating scores for Gaussian Backend for trial file {}".format(trial_file))

            nc.generate_scorefile("scores/GaussianBackend_{}_{}.txt".format(os.path.splitext(os.path.basename(trial_file))[0], timestamp), trial_file, mega_xvec_dict, model, nc.device)


        
if __name__ == '__main__':
    main_GB()
