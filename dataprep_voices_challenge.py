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
    xv_path = 'exp/xvector_nnet_1a'
    xvectors_base_path = os.path.join(base_path, xv_path)
    
    stage = 3
    
    # %% Generate and save training trial keys using SRE SWBD and MX6 datasets
    if stage <= 1:
        data_spk2utt_list = np.asarray([['{}/data/train_16k_combined/male/spk2utt'.format(base_path), '1'],
                                          ['{}/data/train_16k_combined/female/spk2utt'.format(base_path), '1']])
    
        xvector_scp_list = xvector_scp_list = np.asarray(
            ['{}/xvectors_train_16k_combined/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_voices_dev_enrollment_copy/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_voices_dev_test_copy/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_voices_eval_enrollment/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_voices_eval_test/xvector_fullpaths.scp'.format(xvectors_base_path)])
    
    
        train_trial_keys, val_trial_keys = generate_train_trial_keys(data_spk2utt_list, xvector_scp_list, train_and_valid=True, train_ratio=0.95)
        
        # Save the training and validation trials and keys for training NPLDA and other discriminative models
        np.savetxt('trials_and_keys/voxceleb_16k_aug_train_trial_keys_1_10.tsv', train_trial_keys, fmt='%s', delimiter='\t', comments='none')
        np.savetxt('trials_and_keys/voxceleb_16k_aug_validate_trial_keys_1_10.tsv', val_trial_keys, fmt='%s', delimiter='\t', comments='none')
        
        # Save the train and validation xvectors for training a Kaldi PLDA if required
        train_scp_path = '{}/xvectors_voxceleb_aug/train_split/xvector.scp'.format(xvectors_base_path)
        valid_scp_path = '{}/xvectors_voxceleb_aug/valid_split/xvector.scp'.format(xvectors_base_path)
        save_unique_train_valid_xvector_scps(data_spk2utt_list, xvector_scp_list, train_scp_path, valid_scp_path, train_ratio=0.95)
        bp()
        
    # %% Get the Voices dev trials in required format
    if stage <= 2:
        voices_dev_trial_key_file_path = '/home/data/VOICES/interspeech2019Challenge/Development_Data/Speaker_Recognition/sid_dev_lists_and_keys/dev-trial-keys.lst'
        voices_dev_trial_key = np.genfromtxt(voices_dev_trial_key_file_path, dtype=str, skip_header=0)
        voices_dev_trial_key[:,2] = (voices_dev_trial_key[:,2]=='tgt').astype(int).astype(str)
        for i, testid in enumerate(voices_dev_trial_key[:,1]):
            voices_dev_trial_key[i,1] = os.path.splitext(os.path.basename(testid))[0]
        voices_dev_trial_key = voices_dev_trial_key[:,:3]
        
        np.savetxt('trials_and_keys/voices_dev_keys.tsv', voices_dev_trial_key, fmt='%s', delimiter='\t', comments='none')
    
    # %% Make the mega xvector scp with all the xvectors, averaged enrollment xvectors, etc.

    if stage <= 3:
        xvector_scp_list = xvector_scp_list = np.asarray(
            # ['{}/xvectors_train_16k_combined/xvector_fullpaths.scp'.format(xvectors_base_path),
             # '{}/xvectors_voices_dev_enrollment_copy/xvector_fullpaths.scp'.format(xvectors_base_path),
             # '{}/xvectors_voices_dev_test_copy/xvector_fullpaths.scp'.format(xvectors_base_path),
             # '{}/xvectors_voices_eval_enrollment/xvector_fullpaths.scp'.format(xvectors_base_path),
             # '{}/xvectors_voices_eval_test/xvector_fullpaths.scp'.format(xvectors_base_path),
             # '{}/xvectors_sitw_combined/xvector.scp'.format(xvectors_base_path),
             ['{}/xvectors_sitw_eval_enroll/spk_xvector.scp'.format(xvectors_base_path),
              '{}/xvectors_sitw_eval_test/xvector.scp'.format(xvectors_base_path)])
        
        # mega_scp_dict = {}
        mega_xvec_dict = pickle.load(open('xvectors/mega_xvector_voices_voxceleb_16k.pkl','rb'))
        for fx in xvector_scp_list:
            subprocess.call(['sed','-i', 's| {}| {}|g'.format(xv_path, xvectors_base_path), fx])
            with open(fx) as f:
                scp_list = f.readlines()
            scp_dict = {os.path.splitext(os.path.basename(x.split(' ', 1)[0]))[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
            xvec_dict = {os.path.splitext(os.path.basename(x.split(' ', 1)[0]))[0]: kaldi_io.read_vec_flt(x.rstrip('\n').split(' ', 1)[1]) for x in scp_list}
            # mega_scp_dict.update(scp_dict)
            mega_xvec_dict.update(xvec_dict)
        
        # mega_scp = np.c_[np.asarray(list(mega_scp_dict.keys()))[:,np.newaxis], np.asarray(list(mega_scp_dict.values()))]
        
        # np.savetxt('xvectors/mega_xvector_voices_voxceleb_16k.scp', mega_scp, fmt='%s', delimiter=' ', comments='')
        
        pickle.dump(mega_xvec_dict, open('xvectors/mega_xvector_voices_voxceleb_16k.pkl', 'wb'))