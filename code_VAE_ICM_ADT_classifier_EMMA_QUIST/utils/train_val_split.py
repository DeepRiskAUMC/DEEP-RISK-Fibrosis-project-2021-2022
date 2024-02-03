# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:26:26 2022

@author: iremc
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_oversample_EMIDEC(images_train, labels_train, indices, do_oversampling=True):
    train_images, val_images, train_labels,  val_labels, indices_train, indices_val = train_test_split(images_train, labels_train, indices, stratify = labels_train, test_size=0.25, random_state=11)
    print(f" Training has {np.sum(train_labels)} MINF and {np.abs(len(train_labels)-np.sum(train_labels))} NOR")
    print(f" Validation has {np.sum(val_labels)} MINF and {np.abs(len(val_labels)-np.sum(val_labels))} NOR")
    ###oversampling class 0 ###
    if do_oversampling == True:
        difference = np.sum(train_labels) - np.abs(len(train_labels)-np.sum(train_labels)) # difference between 1s(MINF) and 0s (NOR)
        indcs = np.array([i for i,v in enumerate(train_labels) if v == 0])
        rand_subsample = np.random.choice(indcs, difference) # randomly select the indices "difference" times
        
        train_images = np.concatenate([train_images, np.take(train_images, rand_subsample)])
        train_labels=np.concatenate([train_labels, np.take(train_labels, rand_subsample, axis=0)])

        indices_train = np.concatenate([indices_train, np.take(indices_train, rand_subsample)])
    print(f" After oversampling training has {np.sum(train_labels)} MINF and {np.abs(len(train_labels)-np.sum(train_labels))} NOR")
    print(f" After oversampling validation has {np.sum(val_labels)} MINF and {np.abs(len(val_labels)-np.sum(val_labels))} NOR")
    
    return  train_images, val_images, train_labels,  val_labels

def train_val_oversample_AMC(IDS, labels, do_oversampling=True):
    train_IDS, val_IDS, train_labels,  val_labels = train_test_split(IDS, labels, stratify = labels, test_size=0.25, random_state=11)
    print(f" Training has {np.sum(train_labels)} Mortality/ATP and {np.abs(len(train_labels)-np.sum(train_labels))} Non Mortality/ATP")
    print(f" Validation has {np.sum(val_labels)} Mortality/ATP and {np.abs(len(val_labels)-np.sum(val_labels))} Non Mortality/ATP")
    ###oversampling class 0 ###
    if do_oversampling == True:
        difference = np.abs(np.sum(train_labels) - np.abs(len(train_labels)-np.sum(train_labels))) # difference between 1s(MINF) and 0s (NOR)
        indcs = np.array([i for i,v in enumerate(train_labels) if v == 1])
        rand_subsample = np.random.choice(indcs, difference) # randomly select the indices "difference" times
        
        train_IDS = np.concatenate([train_IDS, np.take(train_IDS, rand_subsample)])
        train_labels=np.concatenate([train_labels, np.take(train_labels, rand_subsample, axis=0)])

    print(f" After oversampling training has {np.sum(train_labels)} Mortality/ATP and {np.abs(len(train_labels)-np.sum(train_labels))} Non Mortality/ATP")
    print(f" After oversampling validation has {np.sum(val_labels)} Mortality/ATP and {np.abs(len(val_labels)-np.sum(val_labels))} Non Mortality/ATP")
    
    return train_IDS, val_IDS

