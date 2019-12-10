#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:53:41 2019

@author: shreyasr
"""

import numpy as np
import matplotlib.pyplot as plt


def grep(l,s):
    return [i for i in l if s in i]

def plot_valid_losses(logfile):
    a = np.genfromtxt(logfile,dtype='str',delimiter=',,,,')
    b = grep(a,"Test set: C_det(149):")
    losses = [float(w.split()[-1]) for w in b]
    val_losses = np.array([l for i,l in enumerate(losses) if i%2==0])
    sre18_dev_losses = np.array([l for i,l in enumerate(losses) if i%2==1])
    plt.plot(val_losses)
    plt.plot(sre18_dev_losses)
    
def plot_thresholds(threshold_file):
    a = np.genfromtxt(threshold_file)
    plt.plot(a[2:7])