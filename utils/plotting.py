#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:53:41 2019

@author: shreyasr
"""

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


def grep(l,s):
    return [i for i in l if s in i]

def plot_valid_mincs(logfile, savefile='', nepochs=20):
    a = np.genfromtxt(logfile,dtype='str',delimiter=',,,,')
    b = grep(a,"Test set: C_min(149):")
    losses = [float(w.split()[-1]) for w in b]
    train_losses = np.array([l for i,l in enumerate(losses[1:]) if i%3==0])
    val_losses = np.array([l for i,l in enumerate(losses[1:]) if i%3==1])
    sre18_dev_losses = np.array([l for i,l in enumerate(losses[1:]) if i%3==2])
    plt.figure(figsize=(5,3.5))
    plt.plot(train_losses[:nepochs])
    plt.plot(val_losses[:nepochs])
    plt.plot(sre18_dev_losses[:nepochs])
    x1,x2,y1,y2 = plt.axis()
    # plt.axis((x1,x2,0,1))
    plt.legend(['Vox Train','Vox unseen val. set','VOiCES Dev'])
    plt.xlabel("Epoch #")
    plt.ylabel("minDCF")
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.legend(['Train data Cmin','5% Unseen Validation Cmin','VOICES_Dev Cmin'])
    # title = '\n'.join(wrap("Plot of C_{min}. "+a[1], 60))
    # plt.title(title)
    if savefile:
        plt.savefig("{}_minc.pdf".format(savefile))
    
def plot_valid_softcdets(logfile, savefile=''):
    a = np.genfromtxt(logfile,dtype='str',delimiter=',,,,')
    b = grep(a,"Test set: C_mdl(149):")
    losses = [float(w.split()[-1]) for w in b]
    train_losses = np.array([l for i,l in enumerate(losses[1:]) if i%3==0])
    val_losses = np.array([l for i,l in enumerate(losses[1:]) if i%3==1])
    sre18_dev_losses = np.array([l for i,l in enumerate(losses[1:]) if i%3==2])
    plt.figure(figsize=(8,8))
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.plot(sre18_dev_losses)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,1))
    plt.legend(['Train data Cdet loss','5% Unseen Validation Cdet loss','SRE 2018 Cdet loss'])
    title = '\n'.join(wrap("Plot of C_{det} computed at model threshold (Used for backprop). "+a[1],60))
    plt.title(title)
    if savefile:
        plt.savefig("{}_cdet.png".format(savefile))
    
def plot_thresholds(logfile,threshold_file, savefile=''):
    a = np.genfromtxt(threshold_file)
    b = np.genfromtxt(logfile,dtype='str',delimiter=',,,,')
    x = np.linspace(0,30,len(a))
    # try:
    plt.figure(figsize=(8,8))
    plt.plot(x,a[:,2:4])
    plt.plot(x,a[:,5])
    plt.plot(x,a[:,8])
    plt.legend(["Model Threshold1","Model Threshold2","MinC Threshold1","MinC Threshold2"])
    title = '\n'.join(wrap("Plot of MinC Thresholds for training data. "+b[1],60))
    plt.title(title)
    if savefile:
        plt.savefig("{}_th_train.png".format(savefile))
    plt.figure(figsize=(8,8))
    plt.plot(x,a[:,2:4])
    plt.plot(x,a[:,6])
    plt.plot(x,a[:,9])
    plt.legend(["Model Threshold1","Model Threshold2","MinC Threshold1","MinC Threshold2"])
    title = '\n'.join(wrap("Plot of MinC Thresholds for 5% unseen data. "+b[1],60))
    plt.title(title)
    if savefile:
        plt.savefig("{}_th_unseen.png".format(savefile))
    plt.figure(figsize=(8,8))
    plt.plot(x,a[:,2:4])
    plt.plot(x,a[:,7])
    plt.plot(x,a[:,10])
    plt.legend(["Model Threshold1","Model Threshold2","MinC Threshold1","MinC Threshold2"])
    title = '\n'.join(wrap("Plot of MinC Thresholds for SRE18 dev data. "+b[1],60))
    plt.title(title)
    if savefile:
        plt.savefig("{}_th_sre18dev.png".format(savefile))
    # except:
    #     print("Sh*t happened")
    
def generate_plots():
    logfiles = np.genfromtxt('logs/logs_27122019',dtype='str',delimiter=',,,,')
    thresholds_files = np.genfromtxt('logs/thresholds_27122019',dtype='str',delimiter=',,,,')
    for l,t in zip(logfiles,thresholds_files):
        logfile = 'logs/'+l
        threshold_file = 'logs/'+t
        savefilename = "plots/plt_{}".format(threshold_file.split('_')[-1])
        try:
            plot_valid_softcdets(logfile, savefile=savefilename)
        except:
            print("plot_valid_softcdets failed for {}, {}.".format(logfile,threshold_file))
        try:
            plot_valid_mincs(logfile, savefile=savefilename)
        except:
            print("plot_valid_mincs failed for {}, {}.".format(logfile,threshold_file))
        try:
            plot_thresholds(logfile,threshold_file, savefile=savefilename)
        except:
            print("plot_thresholds failed for {}, {}.".format(logfile,threshold_file))