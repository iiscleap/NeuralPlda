#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:15:48 2020

@author: shreyasr
"""

import re
import numpy as np
import random
import sys
import subprocess
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
import kaldi_io
from pdb import set_trace as bp

def make_same_speaker_list(spk2utt_file, xvector_scp_combined, same_speaker_list_file=None, n_repeats=1, train_and_valid=False,train_ratio=0.95):
    # print("In same speaker list")
    assert train_ratio < 1, "train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    random.seed(2)
    random.shuffle(spk2utt_list)
    uttsperspk = [(a.rstrip('\n').split(' ', 1)[1]).split(' ') for a in spk2utt_list]
    
    train_uttsperspk = uttsperspk[:int(train_ratio * len(uttsperspk))]
    train_same_speaker_list = []
    for repeats in range(n_repeats):
        for utts in train_uttsperspk:
            utts_shuffled = utts.copy()
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in xvector_scp_combined:
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in xvector_scp_combined:
                    continue
                train_same_speaker_list.append([tmp1, tmp2])
    train_same_speaker_list = np.asarray(train_same_speaker_list)
    
    valid_uttsperspk = uttsperspk[int((train_ratio) * len(uttsperspk)):]
    valid_same_speaker_list = []
    for repeats in range(n_repeats):
        for utts in valid_uttsperspk:
            utts_shuffled = utts.copy()
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in xvector_scp_combined:
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in xvector_scp_combined:
                    continue
                valid_same_speaker_list.append([tmp1, tmp2])
    valid_same_speaker_list = np.asarray(valid_same_speaker_list)

    return train_same_speaker_list, valid_same_speaker_list

    if train_and_valid:  # Returns two lists for training and validation
        return train_same_speaker_list, valid_same_speaker_list
    else:
        return train_same_speaker_list + valid_same_speaker_list


def make_diff_speaker_list(spk2utt_file, xvector_scp_combined, diff_speaker_list_file=None, n_repeats=1, train_and_valid=True,
                           train_ratio=0.95):
    # print("In diff speaker list")
    assert train_ratio < 1, "train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    random.seed(2)
    random.shuffle(spk2utt_list)
    spk2utt_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1].split(' ') for x in spk2utt_list}
    spk2utt_keys = list(spk2utt_dict.keys())
    train_keys = spk2utt_keys[:int(train_ratio * len(spk2utt_keys))]
    valid_keys = spk2utt_keys[int(train_ratio * len(spk2utt_keys)):]
    utt2spk_train = []
    utt2spk_valid = []
    for i in train_keys:
        for j in spk2utt_dict[i]:
            utt2spk_train.append([j, i])
    for i in valid_keys:
        for j in spk2utt_dict[i]:
            utt2spk_valid.append([j, i])

    train_diff_speaker_list = []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk_train)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in xvector_scp_combined:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in xvector_scp_combined:
                    continue
                train_diff_speaker_list.append([list(tmp1)[0], list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    bp()
                    break

    valid_diff_speaker_list = []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk_valid)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in xvector_scp_combined:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in xvector_scp_combined:
                    continue
                valid_diff_speaker_list.append([list(tmp1)[0], list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    bp()
                    break
    train_diff_speaker_list = np.asarray(train_diff_speaker_list)
    valid_diff_speaker_list = np.asarray(valid_diff_speaker_list)
    
    if train_and_valid: # Returns two lists for training and validation
        return train_diff_speaker_list, valid_diff_speaker_list
    else:
        return train_diff_speaker_list + valid_diff_speaker_list
    
    
def generate_train_trial_keys(data_spk2utt_list, xvector_scp_list, train_and_valid=True, train_ratio=0.95):

    #    Make sure that each spk2utt in data_spk2utt_list is of same gender, same source, same language, etc. More Matching Metadata --> Better the model training.

    #    Can also specify the num_repeats after the dir name followed with space/tab separation in 2 column format. If not specified, default num_repeats is set to 1.
    
    xvector_scp_combined = {}
    
    for fx in xvector_scp_list:
        with open(fx) as f:
            scp_list = f.readlines()
        scp_dict = {os.path.splitext(os.path.basename(x.split(' ', 1)[0]))[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
        xvector_scp_combined.update(scp_dict)

    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')

    if data_spk2utt_list.ndim == 2:
        num_repeats_list = data_spk2utt_list[:, 1].astype(int)
        data_spk2utt_list = data_spk2utt_list[:, 0]
    elif data_spk2utt_list.ndim == 1:
        num_repeats_list = np.ones(len(data_spk2utt_list)).astype(int)
    else:
        raise("Something wrong here.")


    sampled_list_train = []
    sampled_list_valid = []

    for i, d in enumerate(data_spk2utt_list):
        # print("In for loop get train dataset")
        same_train_list, same_valid_list = make_same_speaker_list(d, xvector_scp_combined, xvector_scp_list, n_repeats = num_repeats_list[i], train_and_valid=True, train_ratio=0.95)
        diff_train_list, diff_valid_list = make_diff_speaker_list(d, xvector_scp_combined, n_repeats = 10*num_repeats_list[i], train_and_valid=True, train_ratio=0.95)
        # bp()
        zeros = np.zeros((diff_train_list.shape[0], 1)).astype(int)
        ones = np.ones((same_train_list.shape[0], 1)).astype(int)
        same_list_with_label_train = np.concatenate((same_train_list, ones), axis=1)
        diff_list_with_label_train = np.concatenate((diff_train_list, zeros), axis=1)
        zeros = np.zeros((diff_valid_list.shape[0], 1)).astype(int)
        ones = np.ones((same_valid_list.shape[0], 1)).astype(int)
        same_list_with_label_valid = np.concatenate((same_valid_list, ones), axis=1)
        diff_list_with_label_valid = np.concatenate((diff_valid_list, zeros), axis=1)
        concat_pair_list_train = np.concatenate((same_list_with_label_train, diff_list_with_label_train))
        concat_pair_list_valid = np.concatenate((same_list_with_label_valid, diff_list_with_label_valid))

        np.random.shuffle(concat_pair_list_train)
        sampled_list_train.extend(concat_pair_list_train)

        np.random.shuffle(concat_pair_list_valid)
        sampled_list_valid.extend(concat_pair_list_valid)
    
    if train_and_valid:
        return sampled_list_train, sampled_list_valid
    else:
        return sampled_list_train + sampled_list_valid
    
def save_unique_train_valid_xvector_scps(data_spk2utt_list, xvector_scp_list, train_scp_path, valid_scp_path, train_ratio=0.95):
    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')

    if data_spk2utt_list.ndim == 2:
        data_spk2utt_list = data_spk2utt_list[:, 0]
        
    xvector_scp_combined = {}
    
    for fx in xvector_scp_list:
        with open(fx) as f:
            scp_list = f.readlines()
        scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
        xvector_scp_combined.update(scp_dict)
    
    train_scp = []
    valid_scp = []
    # bp()
    for i, d in enumerate(data_spk2utt_list):
        with open(d) as f:
            spk2utt_list = f.readlines()
        random.seed(2)
        random.shuffle(spk2utt_list)
        spk2utt_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1].split(' ') for x in spk2utt_list}
        spks = list(spk2utt_dict.keys())
        train_keys = spks[:int(train_ratio * len(spks))]
        valid_keys = spks[int(train_ratio * len(spks)):]
        # bp()
        for i in train_keys:
            for j in spk2utt_dict[i]:
                if j in xvector_scp_combined:
                    train_scp.append([j, xvector_scp_combined[j]])
        for i in valid_keys:
            for j in spk2utt_dict[i]:
                if j in xvector_scp_combined:
                    valid_scp.append([j, xvector_scp_combined[j]])
    train_scp = np.asarray(train_scp)
    valid_scp = np.asarray(valid_scp)
    
    np.savetxt(train_scp_path, train_scp, fmt='%s', delimiter=' ', comments='')
    np.savetxt(valid_scp_path, valid_scp, fmt='%s', delimiter=' ', comments='')

def combine_trials_and_get_loader(trials_key_files_list, id_to_num_dict, batch_size=2048, subset=0):
    datasets = []
    for f in trials_key_files_list:
        t = np.genfromtxt(f, dtype = 'str')
        x1, x2, l = [], [], []
        for tr in t:
            try:
                a, b, c = id_to_num_dict[tr[0]], id_to_num_dict[tr[1]], float(tr[2])
                x1.append(a); x2.append(b); l.append(c)
            except:
                pass
        datasets.append(TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(l)))
    combined_dataset = ConcatDataset(datasets)
    if subset > 0:
        inds = np.arange(len(combined_dataset))[np.random.rand(len(combined_dataset))<subset]
        combined_dataset = Subset(combined_dataset, inds)
    trials_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return trials_loader

def get_trials_loaders_dict(trials_key_files_list, id_to_num_dict, batch_size=2048, subset=0):
    trials_loaders_dict = {}
    for f in trials_key_files_list:
        t = np.genfromtxt(f, dtype = 'str')
        x1, x2, l = [], [], []
        for tr in t:
            try:
                a, b, c = id_to_num_dict[tr[0]], id_to_num_dict[os.path.splitext(tr[1])[0]], float(tr[2])
                x1.append(a); x2.append(b); l.append(c)
            except:
                pass
        dataset = TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(l))
        if subset > 0:
            inds = np.arange(len(dataset))[np.random.rand(len(dataset))<subset]
            dataset = Subset(dataset, inds)
        trials_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trials_loaders_dict[os.path.splitext(os.path.basename(f))[0]] = trials_loader
    return trials_loaders_dict

def load_xvec_from_numbatch(mega_dict, num_to_id_dict, data1, data2, device):
    data1_xvec, data2_xvec = [], []  # torch.tensor([[]]), torch.tensor([[]])
    for i, (d1, d2) in enumerate(zip(data1, data2)):
        data1_xvec_temp, data2_xvec_temp = mega_dict[num_to_id_dict[int(d1)]], mega_dict[num_to_id_dict[int(d2)]]
        data1_xvec.append(data1_xvec_temp)
        data2_xvec.append(data2_xvec_temp)
    tensor_X1 = torch.from_numpy(np.asarray(data1_xvec)).float().to(device)
    tensor_X2 = torch.from_numpy(np.asarray(data2_xvec)).float().to(device)
    return tensor_X1, tensor_X2

def load_xvec_from_idbatch(mega_dict, trials, device):
    data1_xvec, data2_xvec = [], []  # torch.tensor([[]]), torch.tensor([[]])
    for i, (d1, d2) in enumerate(zip(trials[:,0], trials[:,1])):
        data1_xvec_temp, data2_xvec_temp = mega_dict[d1], mega_dict[os.path.splitext(os.path.basename(d2))[0]]
        data1_xvec.append(data1_xvec_temp)
        data2_xvec.append(data2_xvec_temp)
    tensor_X1 = torch.from_numpy(np.asarray(data1_xvec)).float().to(device)
    tensor_X2 = torch.from_numpy(np.asarray(data2_xvec)).float().to(device)
    return tensor_X1, tensor_X2