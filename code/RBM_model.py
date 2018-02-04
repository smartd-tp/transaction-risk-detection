#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 22:14:12 2017

@author: cp
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import datetime 
from sklearn.metrics import roc_auc_score as auc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import roc_curve, auc

df = pd.read_csv("/Users/cp/creditcard.csv")


#############   set up data 
train_ori=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")
test_ori=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")

############




TEST_RATIO = 0.5
train_ori.sort_values('V_Time', inplace = True)
TRA_INDEX = int((1-TEST_RATIO) * train_ori.shape[0])
train_x = train_ori.iloc[:, 2:-1].values
train_y = train_ori.iloc[:, -1].values

test_x = train_ori.iloc[:, 2:-1].values

R_test_x = test_ori.iloc[:, 2:].values
test_y = train_ori.iloc[:, -1].values



cols_mean = []
cols_std = []
for c in range(train_x.shape[1]):
    cols_mean.append(train_x[:,c].mean())
    cols_std.append(train_x[:,c].std())
    train_x[:, c] = (train_x[:, c] - cols_mean[-1]) / cols_std[-1]
    test_x[:, c] =  (test_x[:, c] - cols_mean[-1]) / cols_std[-1]
    
    
from rbm import RBM


####gibbs_sampling_steps??
model = RBM(train_x.shape[1], 20, visible_unit_type='gauss', main_dir='/Users/cp/Documents/GitHub/Fraud-detection-using-deep-learning', model_name='rbm_model',
                 gibbs_sampling_steps=1, learning_rate=0.001, momentum = 0.95, batch_size=1000, num_epochs=10, verbose=1,l2 = 0.01)



model.fit(train_x, validation_set=test_x)


test_cost = model.getFreeEnergy(test_x).reshape(-1)


auc(test_y, test_cost)



fpr, tpr, _ = roc_curve(test_y, test_cost)

fpr_micro, tpr_micro, _ = roc_curve(test_y, test_cost)
roc_auc = auc(fpr_micro, tpr_micro)

plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on val data set')
plt.legend(loc="lower right")
plt.show()






plt.title('Free energy distribution of val set')
plt.xlabel('Free energy')
plt.ylabel('Probabilties')
plt.hist(test_cost[(test_y == 0) & (test_cost < 500)], bins = 100, color='green', normed=1.0, label='Non-Fraud')
plt.hist(test_cost[(test_y == 1) & (test_cost < 500)], bins = 100, color='red', normed=1.0, label = 'Fraud')

plt.legend(loc="upper right")
plt.show()






precisions = []
recalls = []
all_pos = sum(test_y)
for threshold in range(1, 200):
    all_predicted = sum(test_cost > threshold)
    TP = sum((test_cost > threshold) & (test_y == 1))
    
    precisions.append(TP  / all_predicted)
    recalls.append(TP / all_pos)

plt.plot( recalls, label = 'recall')
plt.axvline(100, color = 'red')

plt.title("Recall curve")
plt.xlabel("FE")
plt.ylabel("Recall")
plt.legend()
plt.show()



plt.plot( precisions, label = 'precision')
plt.axvline(100, color = 'red')

plt.title("Precision curve")
plt.xlabel("FE")
plt.ylabel("Precision")
plt.legend()
plt.show()




plt.hist(test_cost[ (test_cost < 500) & (test_cost >20)], bins = 100, color='green', normed=1.0, label='Non-Fraud')








