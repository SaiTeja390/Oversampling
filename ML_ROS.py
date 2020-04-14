
# coding: utf-8

# In[ ]:


from sklearn.datasets import make_multilabel_classification

import torch
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import random


# In[ ]:


def IRLbl(y):
    LABELS = defaultdict(int)
    for sample in y:
        for label in sample:
            LABELS[label] += 1
            
    max_count = max(LABELS.values())
    for key,value in LABELS.items():
        print(f'{key} {LABELS[key]}')
        LABELS[key] = max_count/LABELS[key]
    
        
    return LABELS



def MeanIR(y,ratios=False):
    imb_ratios = IRLbl(y)
    # print(f'\n ke {imb_ratios} \n\n\n\n')
    if ratios:
        return (sum(imb_ratios.values())/len(imb_ratios), imb_ratios)
    else:
        return (sum(imb_ratios.values())/len(imb_ratios))
        



def Label_Bags(X,y):
    " Constructs Bags of samples for every label in the dataset.    A label's Bag contains samples whose label set contains the label."
    assert(len(X) == len(y))
    Bags = defaultdict(list)
    for id, labels in enumerate(y):
        for label in labels:
            Bags[label].append([X[id],y[id]])
    return Bags



def ML_ROS(dataset, p):
    
    (X,y) = dataset
    X = list(X)
    y = list(y)

    samples_to_clone = len(X)*p/(100)
    mean_ir, imb_ratios = MeanIR(y=y, ratios=True)
    label_bags = Label_Bags(X,y)
    
    minBag = {}
    for label in label_bags.keys():
        if imb_ratios[label] > mean_ir:
            minBag[label] = label_bags[label]
    
    while samples_to_clone>0:
        if(minBag):
            for label in list(minBag):
                minBag_i = minBag[label]
                sample = random.sample(minBag_i,1)[0]

                minBag_i.append(sample)
                X.append(sample[0])
                y.append(sample[1])

                mean_ir, imb_ratios = MeanIR(y=y,ratios=True)
                if imb_ratios[label] <= mean_ir:
                    minBag.pop(label)
                    print(f'label {label} popped')
                samples_to_clone-=1
                print(f'{len(X)}, {samples_to_clone} mean_ir {mean_ir}')
        else:
            minBag = {}
            for label in label_bags.keys():
                if imb_ratios[label] > mean_ir:
                    minBag[label] = label_bags[label]
            
            if(minBag is None):
                break
    return X,y
                

