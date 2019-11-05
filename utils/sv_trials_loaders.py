# -*- coding: utf-8 -*-
import numpy as np
import random
import sys
import subprocess
import pickle
import os
import torch
import torch.utils.data as utils


def make_same_speaker_list(spk2utt_file, same_speaker_list_file = None, n_repeats = 1, train_and_valid=False, train_ratio=0.9):
    assert train_ratio<1,"train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    
    uttsperspk = [(a.split(' ',1)[1]).split(' ') for a in spk2utt_list]
    random.shuffle(uttsperspk)
    
    if train_and_valid: #Returns two lists for training and validation
        
        train_uttsperspk = uttsperspk[:int(train_ratio*len(uttsperspk))]
        train_same_speaker_list = []
        for repeats in range(n_repeats):
            for utts in train_uttsperspk:
                utts_shuffled = utts.copy()
                random.shuffle(utts_shuffled)
                for i in range(0,len(utts_shuffled)-1,2):
                    train_same_speaker_list.append([utts_shuffled[i].strip('\n'), utts_shuffled[i+1].strip('\n')])
        train_same_speaker_list = np.asarray(train_same_speaker_list)
        
        valid_uttsperspk = uttsperspk[int((1-train_ratio)*len(uttsperspk)):]
        valid_same_speaker_list = []
        for repeats in range(n_repeats):
            for utts in valid_uttsperspk:
                utts_shuffled = utts.copy()
                random.shuffle(utts_shuffled)
                for i in range(0,len(utts_shuffled)-1,2):
                    valid_same_speaker_list.append([utts_shuffled[i].strip('\n'), utts_shuffled[i+1].strip('\n')])
        valid_same_speaker_list = np.asarray(valid_same_speaker_list)
        
        return train_same_speaker_list, valid_same_speaker_list
    else:    
        same_speaker_list = []
        for repeats in range(n_repeats):
            for utts in uttsperspk:
                utts_shuffled = utts.copy()
                random.shuffle(utts_shuffled)
                for i in range(0,len(utts_shuffled)-1,2):
                    same_speaker_list.append([utts_shuffled[i].strip('\n'), utts_shuffled[i+1].strip('\n')])
        same_speaker_list = np.asarray(same_speaker_list)    
        if type(same_speaker_list_file) is str:
            print("Saving samelist file at {}".format(same_speaker_list_file))
            np.savetxt(same_speaker_list_file, same_speaker_list, delimiter='\t',fmt='%s',comments='')
    
        return same_speaker_list;


def make_diff_speaker_list(utt2spk_file, diff_speaker_list_file = None, n_repeats = 1, train_and_valid=False, train_ratio=0.9):
    assert train_ratio<1,"train_ratio should be less than 1."
    utt2spk = np.genfromtxt(utt2spk_file,dtype = 'str')
    random.shuffle(utt2spk)
    
    if train_and_valid: #Returns two lists for training and validation
        train_utt2spk = utt2spk[:int(train_ratio*len(utt2spk))]
        train_diff_speaker_list = []
        for repeats in range(n_repeats):
            utt2spk_list = list(train_utt2spk)
            random.shuffle(utt2spk_list)
            i=0
            while len(utt2spk_list) >=2 :
                if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                    tmp1 = utt2spk_list.pop()
                    tmp2 = utt2spk_list.pop()
                    train_diff_speaker_list.append([list(tmp1)[0],list(tmp2)[0]])
                    i=0
                else:
                    i=i+1
                    random.shuffle(utt2spk_list)
                    if i==50:
                        break
                
        valid_utt2spk = utt2spk[:int((1-train_ratio)*len(utt2spk))]
        valid_diff_speaker_list = []
        for repeats in range(n_repeats):
            utt2spk_list = list(valid_utt2spk)
            random.shuffle(utt2spk_list)
            i=0
            while len(utt2spk_list) >=2 :
                if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                    tmp1 = utt2spk_list.pop()
                    tmp2 = utt2spk_list.pop()
                    valid_diff_speaker_list.append([list(tmp1)[0],list(tmp2)[0]])
                    i=0
                else:
                    i=i+1
                    random.shuffle(utt2spk_list)
                    if i==50:
                        break
        train_diff_speaker_list = np.asarray(train_diff_speaker_list)
        valid_diff_speaker_list = np.asarray(valid_diff_speaker_list)
        return train_diff_speaker_list, valid_diff_speaker_list
    else:            
        diff_speaker_list = []
        for repeats in range(n_repeats):
            utt2spk_list = list(utt2spk)
            random.shuffle(utt2spk_list)
            i=0
            while len(utt2spk_list) >=2 :
                if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                    tmp1 = utt2spk_list.pop()
                    tmp2 = utt2spk_list.pop()
                    diff_speaker_list.append([list(tmp1)[0],list(tmp2)[0]])
                    i=0
                else:
                    i=i+1
                    random.shuffle(utt2spk_list)
                    if i==50:
                        break
            
        diff_speaker_list = np.asarray(diff_speaker_list)
        if type(diff_speaker_list_file) is str:
            print("Saving difflist file at {}".format(diff_speaker_list_file))
            np.savetxt(diff_speaker_list_file, diff_speaker_list, delimiter='\t',fmt='%s',comments='')
        return diff_speaker_list


def get_spk2xvector(enroll_spk2utt_path,enroll_xvector_path):
    enroll_xvectors=pickle.load(open(enroll_xvector_path, "rb"))
    spk2xvector={}
    with open(enroll_spk2utt_path) as fp:
        for line in fp:
            arr=line.split()
            if len(arr)==2:
                spk2xvector[arr[0]]=enroll_xvectors[arr[1]]
            else:
                a=np.vstack((enroll_xvectors[arr[1]],enroll_xvectors[arr[2]],enroll_xvectors[arr[3]]))
                #print("Stacked shape",a.shape)
                avg_xvector=np.average(a,axis=0)
                #print("Average shape",avg_xvector.shape)
                spk2xvector[arr[0]]=avg_xvector
                    
    return spk2xvector

def get_enrollmodel2xvector(enroll_spk2utt_path,enroll_xvector_path):
    enroll_xvectors=pickle.load(open(enroll_xvector_path, "rb"))
    spk2xvector={}
    with open(enroll_spk2utt_path) as fp:
        for line in fp:
            arr=line.split()
            if len(arr)==2:
                spk2xvector[arr[0]]=enroll_xvectors[arr[1]]
            else:
                a=np.vstack((enroll_xvectors[arr[1]],enroll_xvectors[arr[2]],enroll_xvectors[arr[3]]))
                #print("Stacked shape",a.shape)
                avg_xvector=np.average(a,axis=0)
                #print("Average shape",avg_xvector.shape)
                spk2xvector[arr[0]]=avg_xvector
    return spk2xvector


def kaldivec2numpydict(inArkOrScpFile, outpicklefile=''):
    # This function converts a Kaldi vector file into a dictionary with numpy arrays of the vector as the dictionary values
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

def dataloader_from_list(pair_list, res, batch_size = 64, shuffle=True, train_and_valid=False, train_ratio=0.9):
    x1_array, x2_array, y_array = [],[],[]
    for f in pair_list:
        x1_id,x2_id,label = f[0],f[1],f[2]
        try:    
            x1_vec, x2_vec = res[x1_id], res[x2_id]
        except:
            continue
        x1_array.append(x1_vec)
        x2_array.append(x2_vec)
        y_array.append(int(label))
    tensor_X1 = torch.from_numpy(np.asarray(x1_array)).float() #torch.stack([torch.Tensor(i) for i in x1_array])
    tensor_X2 = torch.from_numpy(np.asarray(x2_array)).float() #torch.stack([torch.Tensor(i) for i in x2_array])
    tensor_Y = torch.from_numpy(np.asarray(y_array)).float() #torch.stack([torch.Tensor(i) for i in y_array]).long()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    if not train_and_valid:
        data_loader = utils.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
        return data_loader
    else:
        dataset_length = len(dataset)
        train_length = int(train_ratio*dataset_length)
        test_length = dataset_length - train_length #int(0.1*dataset_length)
        trainset, valset = utils.random_split(dataset, [train_length,test_length])
        train_loader = utils.DataLoader(trainset,batch_size=batch_size, shuffle=shuffle)
        valid_loader = utils.DataLoader(valset,batch_size=batch_size, shuffle=shuffle)
        return train_loader, valid_loader




def dataloader_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors,batch_size = 256, shuffle=True):
    x1_arr, x2_arr=[], []
    y_arr = []
    with open(trial_file_path) as fp:
       for line in fp:
           x1_id,x2_id,label=line.strip().split()
           try:
               x1_vec=enroll_spk2xvector[x1_id]
               x2_vec=test_xvectors[x2_id]
              # print(type(x1_vec))
              ## print(type(x2_vec))
               #print(len(x1_vec))
               #print(len(x2_vec))
           except:
#               print("Error while loading X-vectors!!!")
               continue
           if label=='target':
               y_arr.append(1)
           elif label=='nontarget': #doing this to catch if there are any other spellings used
               y_arr.append(0)
           x1_arr.append(x1_vec)
           x2_arr.append(x2_vec)
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    tensor_Y = torch.from_numpy(np.asarray(y_arr)).float()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    trial_loader=utils.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    return trial_loader  #tensor_X1, tensor_X2 , np.asarray(y_arr)

def dataset_from_list(pair_list, res, batch_size = 64, shuffle=True):
    x1_array, x2_array, y_array = [],[],[]
    for f in pair_list:
        x1_id,x2_id,label = f[0],f[1],f[2]
        try:    
            x1_vec, x2_vec = res[x1_id], res[x2_id]
        except:
            continue
        x1_array.append(x1_vec)
        x2_array.append(x2_vec)
        y_array.append(int(label))
    tensor_X1 = torch.from_numpy(np.asarray(x1_array)).float() #torch.stack([torch.Tensor(i) for i in x1_array])
    tensor_X2 = torch.from_numpy(np.asarray(x2_array)).float() #torch.stack([torch.Tensor(i) for i in x2_array])
    tensor_Y = torch.from_numpy(np.asarray(y_array)).float() #torch.stack([torch.Tensor(i) for i in y_array]).long()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    return dataset

def dataset_from_trial(trial_file_path, enroll_spk2xvector, test_xvectors,batch_size = 256, shuffle=True):
    x1_arr, x2_arr=[], []
    y_arr = []
    with open(trial_file_path) as fp:
       for line in fp:
           x1_id,x2_id,label=line.strip().split()
           try:
               x1_vec=enroll_spk2xvector[x1_id]
               x2_vec=test_xvectors[x2_id]
              # print(type(x1_vec))
              ## print(type(x2_vec))
               #print(len(x1_vec))
               #print(len(x2_vec))
           except:
#               print("Error while loading X-vectors!!!")
               continue
           if label=='target':
               y_arr.append(1)
           elif label=='nontarget': #doing this to catch if there are any other spellings used
               y_arr.append(0)
           x1_arr.append(x1_vec)
           x2_arr.append(x2_vec)
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    tensor_Y = torch.from_numpy(np.asarray(y_arr)).float()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    return dataset  

def concatenate_datasets(datasets_list,batch_size = 4096):
    combined_dataset = utils.ConcatDataset(datasets_list)
    data_loader = utils.DataLoader(combined_dataset,batch_size=batch_size,shuffle=True)
    return data_loader

def dataloader_from_sre08_10_trial(trials, enroll_model2xvector, test_xvectors, utts_dict, batch_size = 2048, shuffle=True):
    x1_arr, x2_arr=[], []
    y_arr = []

    for line in trials:
        x1_id,x2_id,label=line[0],line[1],line[2]
        try:
           x1_vec=enroll_model2xvector[x1_id]
           x2_vec=test_xvectors[utts_dict[x2_id]]
        except:
#            print("Error while loading X-vectors!!!")
            continue
        if label=='target':
            y_arr.append(1)
        elif label=='nontarget':
            y_arr.append(0)
        x1_arr.append(x1_vec)
        x2_arr.append(x2_vec)
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    tensor_Y = torch.from_numpy(np.asarray(y_arr)).float()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    trial_loader=utils.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    return trial_loader  #tensor_X1, tensor_X2 , np.asarray(y_arr)

def dataloader_from_sre18_dev_vast_trial(trials, enroll_model2xvector, test_xvectors, utts_dict, batch_size = 2048, shuffle=True):
    x1_arr, x2_arr=[], []
    y_arr = []

    for line in trials:
        x1_id,x2_id,label=line[0],line[1],line[2]
        try:
           x1_vec=enroll_model2xvector[x1_id]
           x2_vec=test_xvectors[x2_id]
        except:
#            print("Error while loading X-vectors!!!")
            continue
        if label=='target':
            y_arr.append(1)
        elif label=='nontarget':
            y_arr.append(0)
        x1_arr.append(x1_vec)
        x2_arr.append(x2_vec)
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    tensor_Y = torch.from_numpy(np.asarray(y_arr)).float()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    trial_loader=utils.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    return trial_loader

def dataset_from_sre08_10_trial(trials, enroll_model2xvector, test_xvectors, utts_dict, batch_size = 2048, shuffle=True):
    x1_arr, x2_arr=[], []
    y_arr = []

    for line in trials:
        x1_id,x2_id,label=line[0],line[1],line[2]
        try:
           x1_vec=enroll_model2xvector[x1_id]
           x2_vec=test_xvectors[utts_dict[x2_id]]
        except:
#            print("Error while loading X-vectors!!!")
            continue
        if label=='target':
            y_arr.append(1)
        elif label=='nontarget':
            y_arr.append(0)
        x1_arr.append(x1_vec)
        x2_arr.append(x2_vec)
    
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    tensor_Y = torch.from_numpy(np.asarray(y_arr)).float()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    return dataset

def dataloader_from_trials_list(trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list, batch_size = 2048, shuffle=True):
    x1_arr, x2_arr=[], []
    y_arr = []
    for trial_file_path, enroll_spk2xvector, test_xvectors in zip(trial_file_paths_list, enroll_spk2xvectors_list, test_xvectors_list):
        
        with open(trial_file_path) as fp:
           for line in fp:
               x1_id,x2_id,label=line.strip().split()
               try:
                   x1_vec=enroll_spk2xvector[x1_id]
                   x2_vec=test_xvectors[x2_id]
               except:
                   print("Error while loading X-vectors!!!")
               if label=='target':
                   y_arr.append(1)
               elif label=='nontarget':
                   y_arr.append(0)
               x1_arr.append(x1_vec)
               x2_arr.append(x2_vec)
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    tensor_Y = torch.from_numpy(np.asarray(y_arr)).float()
    dataset = utils.TensorDataset(tensor_X1,tensor_X2,tensor_Y)
    trial_loader=utils.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    return trial_loader  #tensor_X1, tensor_X2 , np.asarray(y_arr)

def get_train_valid_loader(data_dir_list, xvec_list, batch_size=64):
#    Make sure that each data dir in data_dir_list is of same gender, same sorce, same language, etc. More Matching Metadata --> Better the model training.
    
#    Can also specify the num_repeats after the dir name followed with tab separation in 2 column format. If not specified, default num_repeats is set to 1.
    
#   xvec_list must contain the ark or scp files of all the utts present in the dirs of data_dir_list
  
    if type(data_dir_list)==str:
        data_dir_list = np.genfromtxt(data_dir_list, dtype='str', delimiter='\t')
    
    if data_dir_list.ndim == 2:
        num_repeats_list = data_dir_list[:,1].astype(int)
        data_dir_list = data_dir_list[:,0]
        
    if type(xvec_list)==str:
        xvec_list = np.genfromtxt()

    mega_list = []
    
    for i,d in enumerate(data_dir_list):
        same_list = make_same_speaker_list(os.path.join(d,'spk2utt'),n_repeats=num_repeats_list[i], train_and_valid=False)
        diff_list = make_diff_speaker_list(os.path.join(d,'utt2spk'),n_repeats=10*num_repeats_list[i], train_and_valid=False)
        
        zeros = np.zeros((diff_list.shape[0],1)).astype(int)
        ones = np.ones((same_list.shape[0],1)).astype(int)
        same_list_with_label = np.concatenate((same_list,ones),axis=1)
        diff_list_with_label = np.concatenate((diff_list,zeros),axis=1)
        concat_pair_list = np.concatenate((same_list_with_label,diff_list_with_label))
        
        np.random.shuffle(concat_pair_list)
        mega_list.extend(concat_pair_list)
    
    xvec_dict = {}
    for arkOrScpFile in xvec_list:
        res = kaldivec2numpydict(arkOrScpFile)
        xvec_dict.update(res)
    
    print(len(mega_list))
    train_loader, valid_loader = dataloader_from_list(mega_list, xvec_dict, batch_size = batch_size, shuffle=True, train_and_valid=True, train_ratio=0.9)
    return train_loader, valid_loader

def get_train_dataset(data_dir_list, xvec_list, batch_size=64):
#    Make sure that each data dir in data_dir_list is of same gender, same sorce, same language, etc. More Matching Metadata --> Better the model training.
    
#    Can also specify the num_repeats after the dir name followed with tab separation in 2 column format. If not specified, default num_repeats is set to 1.
    
#   xvec_list must contain the ark or scp files of all the utts present in the dirs of data_dir_list
    
    if type(data_dir_list)==str:
        data_dir_list = np.genfromtxt(data_dir_list, dtype='str', delimiter='\t')
    
    if data_dir_list.ndim == 2:
        num_repeats_list = data_dir_list[:,1].astype(int)
        data_dir_list = data_dir_list[:,0]
        
    if type(xvec_list)==str:
        xvec_list = np.genfromtxt()
        
    mega_list = []
    
    for i,d in enumerate(data_dir_list):
        same_list = make_same_speaker_list(os.path.join(d,'spk2utt'),n_repeats=num_repeats_list[i], train_and_valid=False)
        diff_list = make_diff_speaker_list(os.path.join(d,'utt2spk'),n_repeats=10*num_repeats_list[i], train_and_valid=False)
        
        zeros = np.zeros((diff_list.shape[0],1)).astype(int)
        ones = np.ones((same_list.shape[0],1)).astype(int)
        same_list_with_label = np.concatenate((same_list,ones),axis=1)
        diff_list_with_label = np.concatenate((diff_list,zeros),axis=1)
        concat_pair_list = np.concatenate((same_list_with_label,diff_list_with_label))
        
        np.random.shuffle(concat_pair_list)
        mega_list.extend(concat_pair_list)
    
    xvec_dict = {}
    for arkOrScpFile in xvec_list:
        res = kaldivec2numpydict(arkOrScpFile)
        xvec_dict.update(res)
    
    print(len(mega_list))
    train_set = dataset_from_list(mega_list, xvec_dict, batch_size = batch_size, shuffle=True)
    return train_set


def xv_pairs_from_trial(trials_file, enroll_spk2xvector, test_xvectors):
    # Returns tensors containing xvector features of the trial pair
    x1_arr, x2_arr=[], []
    fp = np.genfromtxt(trials_file, dtype='str')
    if 'model' in fp[0,0]:
        fp = fp[1:]
    for line in fp:
        x1_id,x2_id=line[0],line[1]
        try:
            x1_vec=enroll_spk2xvector[x1_id]
            x2_vec=test_xvectors[x2_id]
        except:
            print("Error while loading X-vectors!!!")
        x1_arr.append(x1_vec)
        x2_arr.append(x2_vec)
    tensor_X1 = torch.from_numpy(np.asarray(x1_arr)).float() 
    tensor_X2 = torch.from_numpy(np.asarray(x2_arr)).float() 
    return tensor_X1, tensor_X2

def generate_scores_in_batches(score_filename, device, trials_file, x1, x2, model, scores=None):
    # To reduce memory usage on CPU, scores are generated in batches and then concatenated
    
    model = model.cpu()
    batch_size = 1024
    iters = x1.shape[0]//batch_size
    x1 = x1.cpu()
    x2 = x2.cpu()
    S = torch.tensor([])
    model = model.eval()
    with torch.no_grad():
        if scores is None:
            for i in range(iters):
                x1_b = x1[i*batch_size:i*batch_size + batch_size]
                x2_b = x2[i*batch_size:i*batch_size + batch_size]
                S_b = model.forward(x1_b,x2_b)
                S = torch.cat((S,S_b))
            x1_b = x1[iters*batch_size:]
            x2_b = x2[iters*batch_size:]
            S_b = model.forward(x1_b,x2_b)
            S = torch.cat((S,S_b))
            scores = np.asarray(S.detach()).astype(str)
        else:
            scores = np.asarray(scores).astype(str)
    trials = np.genfromtxt(trials_file, dtype='str')
    header = '\t'.join(trials[0])+'\tLLR'
    np.savetxt(score_filename, np.c_[trials[1:],scores], header=header, fmt='%s', delimiter='\t', comments='')
    model=model.to(device)

def generate_scores_from_net(score_filename, device, trials_file, x1, x2, model, scores=None):
    model = model.cpu()
    x1 = x1.cpu()
    x2 = x2.cpu()
    model = model.eval()
    if scores is None:
        S = model.forward(x1,x2)
        scores = np.asarray(S.detach()).astype(str)
    else:
        scores = np.asarray(scores).astype(str)
    trials = np.genfromtxt(trials_file, dtype='str')
    header = '\t'.join(trials[0])+'\tLLR'
    np.savetxt(score_filename, np.c_[trials[1:],scores], header=header, fmt='%s', delimiter='\t', comments='')
    model=model.to(device)