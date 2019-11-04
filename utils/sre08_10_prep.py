#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:23:21 2019

@author: shreyasr
"""

import numpy as np
import os
import subprocess
import pickle
import re
from pdb import set_trace as bp

def kaldivec2numpydict(inArkOrScpFile, outpicklefile=''):
    #logging.debug('kaldi text file to numpy array: {}'.format(textfile))
    if os.path.splitext(inArkOrScpFile)[1] == '.scp':
        fin = subprocess.check_output(["copy-vector", "scp:{}".format(inArkOrScpFile),"ark,t:-"])
    else: #Assuming ARK
        fin = subprocess.check_output(["copy-vector", "ark:{}".format(inArkOrScpFile),"ark,t:-"])
    res = {}
    fin = fin.decode("utf-8").split('\n')
    while '' in fin:
        fin.remove('')
    for line in fin:
        splitted = line.strip().split()
        res[splitted[0]] = np.asarray(splitted[2:-1]).astype(float)
    if outpicklefile:
        with open(outpicklefile,'wb') as f:
            pickle.dump(res,f)
    else:
        return res

def get_enrollmodel2xvector(model_key_dict, all_utts_dict, enroll_xvectors):
#    enroll_xvectors=pickle.load(open(enroll_xvector_path, "rb"))
    enrollmodel2xvector={}
    for modelid, uttids in model_key_dict.items():
        tmparr = []
        for u in uttids:
            try:
                tmparr.append(enroll_xvectors[all_utts_dict[u]])
            except:
                try:
                    tmparr.append(enroll_xvectors[str(u)])
                except:
                    pass
        if tmparr != []:
            enrollmodel2xvector[modelid] = np.mean(np.asarray(tmparr),axis=0)
    return enrollmodel2xvector

def get_sre08_trials_etc():
    xvector_scp_file = '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre08/xvector.scp'
    trials_key_file = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/NIST_SRE08_ALL.trial.key'
    #model_ids_file = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/model_ids.lst'
    model_key_file = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/model-keys/NIST_SRE08_ALL.model.key'
    
    
    xvector_scp = np.genfromtxt(xvector_scp_file,dtype=str)
    trials_key = np.genfromtxt(trials_key_file,dtype=str,delimiter=',')
    model_key = np.genfromtxt(model_key_file,dtype=str,delimiter=',')
    
    all_utts_dict = {(w.split('_')[-2]+'_'+w.split('_')[-1].lower()):w for w in xvector_scp[:,0]}
    
#    n_enrolls = np.unique([len(w.split()) for w in model_key[:,2]])
    model_key_dict = {w[0]:w[2].replace(':','_').split() for w in model_key}
    
    trials = [[w[0], w[1]+'_'+w[2], w[3]] for w in trials_key]
    
    enroll_xvectors = kaldivec2numpydict('/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre08/xvector_fullpaths.scp')
    enrollmodel2xvector = get_enrollmodel2xvector(model_key_dict, all_utts_dict, enroll_xvectors)
    return trials,enroll_xvectors, enrollmodel2xvector,all_utts_dict

def get_sre18_dev_vast_trials_etc():
#    xvector_scp_file = '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_train_8k/xvector.scp'
    trials_key_file = '/home/data1/prachis/SRE_19/PLDA_DNN_Vast/vast_feature_extraction/data/sre18_dev_test/trials_vast'
    #model_ids_file = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/model_ids.lst'
    model_key_file = '/home/data1/prachis/SRE_19/PLDA_DNN_Vast/vast_feature_extraction/data/sre18_dev_enrollment_flac/spk2utt'

#    xvector_scp = np.genfromtxt(xvector_scp_file,dtype=str)
    trials = np.genfromtxt(trials_key_file,dtype=str,delimiter=' ')
#    model_key = np.genfromtxt(model_key_file,dtype=str,delimiter='')
    with open(model_key_file,'r') as f:
        model_key = f.readlines()
        model_key = [w.strip().split() for w in model_key]
        
    all_utts_dict = None
    
#    all_utts_dict = {(w.split('_')[-2]+'_'+w.split('_')[-1].lower()):w for w in xvector_scp[:,0]}
    
#    n_enrolls = np.unique([len(w.split()) for w in model_key[:,2]])
    model_key_dict = {w[0]:w[1:] for w in model_key}
    enroll_xvectors = kaldivec2numpydict('/home/data1/prachis/SRE_19/PLDA_DNN_Vast/vast_feature_extraction/exp/xvectors_sre18_dev_enrollment_flac/xvector_fullpaths.scp')
    enrollmodel2xvector = get_enrollmodel2xvector(model_key_dict, all_utts_dict, enroll_xvectors)
    test_xvectors =  kaldivec2numpydict('/home/data1/prachis/SRE_19/PLDA_DNN_Vast/vast_feature_extraction/exp/xvectors_sre18_dev_test/xvector_fullpaths.scp')
    return trials,enroll_xvectors, enrollmodel2xvector,test_xvectors


def get_sre10_trials_etc():
    xvector_scp_file_10 = '/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre10/xvector.scp'
    trials_key_file_10 = '/home/data/SRE10/export/corpora5/SRE/SRE2010/eval/keys/NIST_SRE10_ALL.trial.key'
    #model_ids_file_10 = '/home/data/SRE08_TEST/export/corpora5/LDC/LDC2011S08/data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/model_ids.lst'
    model_key_file_10 = '/home/data/SRE10/export/corpora5/SRE/SRE2010/eval/train/NIST_SRE10_ALL.model.key'    
    xvector_scp_10 = np.genfromtxt(xvector_scp_file_10,dtype=str)
    trials_key_10 = np.genfromtxt(trials_key_file_10,dtype=str,delimiter=',')
    trials_key_10_subset = trials_key_10[np.random.rand(len(trials_key_10))<0.12]
    model_key_10 = np.asarray([re.split(' m | f ',w.strip()) for w in open(model_key_file_10,'r').readlines()]) #np.genfromtxt(model_key_file_10,dtype=str,delimiter=[' m ',' f '])
    
    all_utts_dict_10 = {(w.split('_')[-2]+'_'+w.split('_')[-1]):w for w in xvector_scp_10[:,0]}
    
#    n_enrolls_10 = np.unique([len(w.split()) for w in model_key_10[:,1]])
    model_key_dict_10 = {w[0]:[os.path.basename(x) for x in w[1].replace('.sph','').replace(':','_').split()] for w in model_key_10}
    
    trials_10 = [[w[0], w[1]+'_'+(w[2]).upper(), w[3]] for w in trials_key_10_subset]
    
    enroll_xvector_10 = kaldivec2numpydict('/home/data2/SRE2019/prashantk/voxceleb/v2/exp/xvector_nnet_1a/xvectors_sre10/xvector_fullpaths.scp')
    
    enrollmodel2xvector_10 = get_enrollmodel2xvector(model_key_dict_10, all_utts_dict_10, enroll_xvector_10)
    return trials_10, enroll_xvector_10, enrollmodel2xvector_10, all_utts_dict_10