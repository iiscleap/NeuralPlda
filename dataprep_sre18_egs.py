#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:44:26 2020

@author: shreyasr
"""

import numpy as np
import random
import pickle
import subprocess
import re
import os
import sys
import kaldi_io

from pdb import set_trace as bp

from utils.sv_trials_loaders import generate_train_trial_keys, save_unique_train_valid_xvector_scps



if __name__=='__main__':
    
    base_path = '/home/data2/SRE2019/prashantk/voxceleb/v3'
    xvectors_base_path = os.path.join(base_path,'exp/xvector_nnet_sre18_3')
    
    stage = 1
    
    # %% Generate and save training trial keys using SRE SWBD and MX6 datasets
    if stage <= 1:
        data_spk2utt_list = np.asarray([['{}/egs/male/spk2utt'.format(xvectors_base_path), '5'],
                                        ['{}/egs/female/spk2utt'.format(xvectors_base_path), '5']])

    
        xvector_scp_list = np.asarray(
            ['{}/egs/egs.scp'.format(xvectors_base_path)])
    
    
        train_trial_keys, val_trial_keys = generate_train_trial_keys(data_spk2utt_list, xvector_scp_list, train_and_valid=True, train_ratio=0.95)
        
        # Save the training and validation trials and keys for training NPLDA and other discriminative models
        np.savetxt('trials_and_keys/sre18_egs_train_trial_keys.tsv', train_trial_keys, fmt='%s', delimiter='\t', comments='none')
        np.savetxt('trials_and_keys/sre18_egs_validate_trial_keys.tsv', val_trial_keys, fmt='%s', delimiter='\t', comments='none')
        sys.exit()
        # Save the train and validation xvectors for training a Kaldi PLDA if required
        # train_scp_path = '{}/xvectors_swbd_sre04to10_mx6/train_split/xvector.scp'.format(xvectors_base_path)
        # valid_scp_path = '{}/xvectors_swbd_sre04to10_mx6/valid_split/xvector.scp'.format(xvectors_base_path)
        # save_unique_train_valid_xvector_scps(data_spk2utt_list, xvector_scp_list, train_scp_path, valid_scp_path, train_ratio=0.95)

    # %% Make SRE 18 dev and eval trial keys in required format using existing trial keys
    
    if stage <= 2:
        sre18_dev_trial_key_file_path = "/home/data/SRE2019/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv"
        sre18_dev_trial_key = np.genfromtxt(sre18_dev_trial_key_file_path, dtype=str, skip_header=1)
        sre18_dev_trial_key[:,2] = (sre18_dev_trial_key[:,3]=='target').astype(int).astype(str)
        for i, testid in enumerate(sre18_dev_trial_key[:,1]):
            sre18_dev_trial_key[i,1] = os.path.splitext(os.path.basename(testid))[0]
        sre18_dev_trial_key = sre18_dev_trial_key[:,:3]
        
        np.savetxt('trials_and_keys/sre18_dev_keys.tsv', sre18_dev_trial_key, fmt='%s', delimiter='\t', comments='none')
    
        sre18_eval_trial_key_file_path = "/home/data/SRE2019/LDC2019E59/eval/docs/sre18_eval_trial_key.tsv"
        sre18_eval_trial_key = np.genfromtxt(sre18_eval_trial_key_file_path, dtype=str, skip_header=1)
        sre18_eval_trial_key[:,2] = (sre18_eval_trial_key[:,3]=='target').astype(int).astype(str)
        for i, testid in enumerate(sre18_eval_trial_key[:,1]):
            sre18_eval_trial_key[i,1] = os.path.splitext(os.path.basename(testid))[0]
        sre18_eval_trial_key = sre18_eval_trial_key[:,:3]
        
        np.savetxt('trials_and_keys/sre18_eval_keys.tsv', sre18_dev_trial_key, fmt='%s', delimiter='\t', comments='none')
        
        sys.exit()

    # %% Get SRE 2008 trials in required format
    
    if stage <= 3:
        xvector_scp_file = '{}/xvectors_sre08/xvector_fullpaths.scp'.format(xvectors_base_path)
        trials_key_file = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/NIST_SRE08_ALL.trial.key'
        model_key_file = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/model-keys/NIST_SRE08_ALL.model.key'
        xvector_scp = np.genfromtxt(xvector_scp_file,dtype=str)
        xvector_scp_dict = dict(zip(xvector_scp[:,0], xvector_scp[:,1]))
        trials = np.genfromtxt(trials_key_file,dtype=str,delimiter=',')
        model_key = np.genfromtxt(model_key_file,dtype=str,delimiter=',')
        all_utts_dict = {(w.split('_')[-2]+'_'+w.split('_')[-1].lower()):w for w in xvector_scp[:,0]}
        model_key_dict = {w[0]:[all_utts_dict[a] for a in w[2].replace(':','_').split() if a in all_utts_dict] for w in model_key}
        model_key_dict = {k:v for k,v in model_key_dict.items() if len(v)>0}
        enroll_spk2utt = np.sort(["{} {}".format(k,' '.join(v)) for k,v in model_key_dict.items()])
        trials = [[w[0], w[1]+'_'+w[2], w[3]] for w in trials]
        trials_key_SRE08 = [[w[0], all_utts_dict[w[1]],str(int(w[2]=='target'))] for w in trials if w[1] in all_utts_dict]
        
        np.savetxt('{}/xvectors_sre08/enroll_spk2utt'.format(xvectors_base_path), enroll_spk2utt, fmt='%s', delimiter='\t', comments='none')
        subprocess.call(['ivector-mean', 'ark:{}/xvectors_sre08/enroll_spk2utt'.format(xvectors_base_path), 'scp:{}/xvectors_sre08/xvector_fullpaths.scp'.format(xvectors_base_path), 'ark,scp:{}/xvectors_sre08/enroll_xvector.ark,{}/xvectors_sre08/enroll_xvector.scp'.format(xvectors_base_path,xvectors_base_path)])
        np.savetxt('trials_and_keys/sre08_eval_trial_keys.tsv', trials_key_SRE08, fmt='%s', delimiter='\t', comments='none')

    # %% Get SRE 2010 trials in required format
    if stage <= 4:
        xvector_scp_file = '{}/xvectors_sre10/xvector_fullpaths.scp'.format(xvectors_base_path)
        trials_key_file = '/home/data/SRE10/export/corpora5/SRE/SRE2010/eval/keys/NIST_SRE10_ALL.trial.key'
        model_key_file = '/home/data/SRE10/export/corpora5/SRE/SRE2010/eval/train/NIST_SRE10_ALL.model.key'
        xvector_scp_10 = np.genfromtxt(xvector_scp_file, dtype=str)
        xvector_scp_10_dict = dict(zip(xvector_scp_10[:,0], xvector_scp_10[:,1]))
        trials_key_10 = np.genfromtxt(trials_key_file, dtype=str, delimiter=',')
        trials_key_10_subset = trials_key_10[np.random.rand(len(trials_key_10))<0.12]
        model_key_10 = np.asarray([re.split(' m | f ',w.strip()) for w in open(model_key_file,'r').readlines()])
        all_utts_dict_10 = {('_'.join(w.split('_')[-2:])):w for w in xvector_scp_10[:,0]}
        model_key_dict_10 = {w[0]:[os.path.basename(x) for x in w[1].replace('.sph','').replace(':','_').split()] for w in model_key_10}
        model_key_dict_10 = {k:[all_utts_dict_10[w] for w in v if w in all_utts_dict_10] for k,v in model_key_dict_10.items()}
        model_key_dict_10 = {k:v for k,v in model_key_dict_10.items() if len(v)>0}
        enroll_spk2utt = np.sort(["{} {}".format(k,' '.join(v)) for k,v in model_key_dict_10.items()])
        
        np.savetxt('{}/xvectors_sre10/enroll_spk2utt'.format(xvectors_base_path), enroll_spk2utt, fmt='%s', delimiter='\t', comments='none')
        subprocess.call(['ivector-mean', 'ark:{}/xvectors_sre10/enroll_spk2utt'.format(xvectors_base_path), 'scp:{}/xvectors_sre10/xvector_fullpaths.scp'.format(xvectors_base_path), 'ark,scp:{}/xvectors_sre10/enroll_xvector.ark,{}/xvectors_sre10/enroll_xvector.scp'.format(xvectors_base_path,xvectors_base_path)])
        
        trials = [[w[0], w[1]+'_'+(w[2]).upper(), w[3]] for w in trials_key_10]
        trials_subset = [[w[0], w[1]+'_'+(w[2]).upper(), w[3]] for w in trials_key_10_subset]
        trials_key_SRE10 = [[w[0], all_utts_dict_10[w[1]],str(int(w[2]=='target'))] for w in trials if w[1] in all_utts_dict_10]
        trials_key_subset_SRE10 = [[w[0], all_utts_dict_10[w[1]],str(int(w[2]=='target'))] for w in trials_subset if w[1] in all_utts_dict_10]
        
        np.savetxt('trials_and_keys/sre10_eval_trial_keys.tsv', trials_key_SRE10, fmt='%s', delimiter='\t', comments='none')
        np.savetxt('trials_and_keys/sre10_eval_trial_keys_subset.tsv', trials_key_subset_SRE10, fmt='%s', delimiter='\t', comments='none')
        
    # %% Make the mega xvector scp with all the xvectors, averaged enrollment xvectors, etc.
        
    if stage <= 5:
        xvector_scp_list = np.asarray(
            ['{}/xvectors_swbd/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_mx6/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre16_eval_enrollment/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre16_eval_test/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre16_eval_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_dev_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_dev_test/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_dev_enrollment/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_test/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_enrollment/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre19_eval_test/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre19_eval_enrollment/xvector_fullpaths.scp'.format(xvectors_base_path),
             '{}/xvectors_sre19_eval_enrollment/spk_xvector.scp'.format(xvectors_base_path)])
        
        mega_scp_dict = {}
        mega_xvec_dict = {}
        for fx in xvector_scp_list:
            subprocess.call(['sed','-i', 's| exp/xvector_nnet_1a| {}|g'.format(xvectors_base_path), fx])
            with open(fx) as f:
                scp_list = f.readlines()
            scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
            xvec_dict = {x.split(' ', 1)[0]: kaldi_io.read_vec_flt(x.rstrip('\n').split(' ', 1)[1]) for x in scp_list}
            mega_scp_dict.update(scp_dict)
            mega_xvec_dict.update(xvec_dict)
        
        mega_scp = np.c_[np.asarray(list(mega_scp_dict.keys()))[:,np.newaxis], np.asarray(list(mega_scp_dict.values()))]
        
        np.savetxt('xvectors/mega_xvector_voxceleb_8k.scp', mega_scp, fmt='%s', delimiter=' ', comments='')
        
        pickle.dump(mega_xvec_dict, open('xvectors/mega_xvector_voxceleb_8k.pkl', 'wb'))