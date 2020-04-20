#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:11:05 2020

@author: shreyasr
"""


import torch
import numpy as np
import os
import random
import pickle
from utils.NpldaConf import NpldaConf

# %% Set config and stuff here

configfile = 'conf/voices_config.cfg'
timestamp = '1586347612'
epoch = '13'

# %% Main 

nc = NpldaConf(configfile)

torch.manual_seed(nc.seed)
np.random.seed(nc.seed)
random.seed(nc.seed)

mega_xvec_dict = pickle.load(open(nc.mega_xvector_pkl, 'rb'))
num_to_id_dict = {i: j for i, j in enumerate(list(mega_xvec_dict))}
id_to_num_dict = {v: k for k, v in num_to_id_dict.items()}

if not torch.cuda.is_available():
    nc.device='cpu'
device = torch.device(nc.device)
    
model = pickle.load(open("models/NPLDA_{}_{}.pt".format(epoch, timestamp),'rb'))

for trial_file in nc.test_trials_list:
    print("Generating scores for Epoch {} with trial file {}".format(epoch, trial_file))

    nc.generate_scorefile("scores/kaldipldanet_epoch{}_{}_{}_scores.txt".format(epoch, os.path.splitext(os.path.basename(trial_file))[0], timestamp), trial_file, mega_xvec_dict, model, device, 5*nc.batch_size)

