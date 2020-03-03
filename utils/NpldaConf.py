#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:15:25 2020

@author: shreyasr
"""

import configparser as cp
from utils.scorefile_generator import generate_voices_scores, generate_sre_scores
    
class NpldaConf:
    def __init__(self, configfile):
        config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
        try:
            config.read(configfile)
        except:
            raise IOError('Oh No! :-( Something is wrong with the config file.')
        self.training_data_trials_list = config['Paths']['training_data_trials_list'].split(',')
        self.validation_trials_list = config['Paths']['validation_trials_list'].split(',')
        self.test_trials_list = config['Paths']['test_trials_list'].split(',')
        self.mega_xvector_scp = config['Paths']['mega_xvector_scp']
        self.mega_xvector_pkl = config['Paths']['mega_xvector_pkl']
        self.meanvec = config['Paths']['meanvec']
        self.transformmat = config['Paths']['transformmat']
        self.kaldiplda = config['Paths']['kaldiplda']
        self.xvector_dim = int(config['NPLDA']['xvector_dim'])
        self.layer1_LDA_dim = int(config['NPLDA']['layer1_LDA_dim'])
        self.layer2_PLDA_spkfactor_dim = int(config['NPLDA']['layer2_PLDA_spkfactor_dim'])
        self.initialization = config['NPLDA']['initialization']
        self.device = config['NPLDA']['device']
        self.seed = int(config['NPLDA']['seed'])
        self.alpha = float(config['NPLDA']['alpha'])
        self.loss = config['Training']['loss']
        self.target_probs = config['Training']['target_probs'].split(',')
        self.beta = [(1-float(pt))/float(pt) for pt in self.target_probs]
        self.batch_size = int(config['Training']['batch_size'])
        self.n_epochs = int(config['Training']['n_epochs'])
        self.lr = float(config['Training']['lr'])
        self.heldout_set_for_lr_decay = config['Training']['heldout_set_for_lr_decay']
        self.heldout_set_for_th_init = config['Training']['heldout_set_for_th_init']
        self.log_interval = int(config['Logging']['log_interval'])
        if config['Scoring']['scorefile_format'] == 'sre':
            self.generate_scorefile = generate_sre_scores
        else:
            self.generate_scorefile = generate_voices_scores