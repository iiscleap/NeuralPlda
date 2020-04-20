#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:51:09 2020

@author: shreyasr
"""

import numpy as np
import random
import pickle
import subprocess
import re
import os
import kaldi_io

from pdb import set_trace as bp

from utils.sv_trials_loaders import generate_train_trial_keys, save_unique_train_valid_xvector_scps



if __name__=='__main__':
    
    base_path = '/home/data2/SRE2019/prashantk/voxceleb/v1'
    xvectors_base_path = os.path.join(base_path,'exp/xvector_nnet_1a')
    
    stage = 1
    
    # %% Generate and save training trial keys using SRE SWBD and MX6 datasets
    if stage <= 1:
        data_spk2utt_list = np.asarray([['{}/data/sdsv_challenge_task2_train/male/spk2utt'.format(base_path), '100'],
                                          ['{}/data/sdsv_challenge_task2_train/female/spk2utt'.format(base_path), '80']])
    
        xvector_scp_list = xvector_scp_list = np.asarray(
            ['{}/xvectors_sdsv_challenge_task2_train/xvector.scp'.format(xvectors_base_path)])
    
    
        train_trial_keys, val_trial_keys = generate_train_trial_keys(data_spk2utt_list, xvector_scp_list, train_and_valid=True, train_ratio=0.95)
        
        # Save the training and validation trials and keys for training NPLDA and other discriminative models
        np.savetxt('trials_and_keys/sdsvc_train_trial_keys_100_80.tsv', train_trial_keys, fmt='%s', delimiter='\t', comments='none')
        np.savetxt('trials_and_keys/sdsvc_validate_trial_keys_100_80.tsv', val_trial_keys, fmt='%s', delimiter='\t', comments='none')
        
        # Save the train and validation xvectors for training a Kaldi PLDA if required
        train_scp_path = '{}/xvectors_sdsvc/train_split/xvector.scp'.format(xvectors_base_path)
        valid_scp_path = '{}/xvectors_sdsvc_aug/valid_split/xvector.scp'.format(xvectors_base_path)
        subprocess.call(['mkdir', '-p', os.path.dirname(train_scp_path)])
        subprocess.call(['mkdir', '-p', os.path.dirname(valid_scp_path)])
        save_unique_train_valid_xvector_scps(data_spk2utt_list, xvector_scp_list, train_scp_path, valid_scp_path, train_ratio=0.9)
        
        exit()
        
    # %% Make the mega xvector scp with all the xvectors, averaged enrollment xvectors, etc.

    if stage <= 2:
        mega_xvec_dict = pickle.load(open('xvectors/mega_xvector_voices_voxceleb_16k.pkl', 'rb'))
        xvector_scp_list = xvector_scp_list = np.asarray(
            ['{}/xvectors_sdsv_challenge_task2_train/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sdsv_challenge_task2.enroll/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sdsv_challenge_task2.enroll/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sdsv_challenge_task2.test/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sdsv_challenge_task2.test/spk_xvector.scp'.format(xvectors_base_path)])

        for fx in xvector_scp_list:
            subprocess.call(['sed','-i', 's| exp/xvector_nnet_1a| {}|g'.format(xvectors_base_path), fx])
            with open(fx) as f:
                scp_list = f.readlines()
            xvec_dict = {x.split(' ', 1)[0]: kaldi_io.read_vec_flt(x.rstrip('\n').split(' ', 1)[1]) for x in scp_list}
            mega_xvec_dict.update(xvec_dict)
      
        pickle.dump(mega_xvec_dict, open('xvectors/mega_xvector_voxceleb_16k_sdsvc.pkl', 'wb'))
        