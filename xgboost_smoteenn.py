#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:29:04 2018

@author: cp
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import f1_score  ,confusion_matrix
from sklearn import  metrics   #Additional     scklearn functions
train = pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")
train_x = train.iloc[:,1:-1]
train_y = train['Label']
sm = SMOTEENN(ratio={1:8000})
X_resampled, y_resampled = sm.fit_sample(np.array(train_x), np.array(train_y))
pred = pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
target = 'Label'
IDcol = 'ID'
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
        #Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
        #Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        pred_y = alg.predict(pred[predictors])
        ori_pred = alg.predict(train[predictors])
        output = pd.concat([pred,pd.DataFrame({'Label':pred_y})],axis = 1)
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
        #Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        print("f1_score:%f" % f1_score(np.array(dtrain[target]),dtrain_predictions))
        print(confusion_matrix(np.array(dtrain[target]),dtrain_predictions))
        print("f1_score_ori:%f" % f1_score(np.array(train[target]),ori_pred))
        print("ori_confuse:")
        print(confusion_matrix(np.array(train[target]),ori_pred))
        print(pd.DataFrame(pred_y)[0].value_counts())
        return [output,ori_pred]

predictors = [x for x in train.columns if x not in [target,IDcol]]
train_resample_x = pd.DataFrame(X_resampled,columns = predictors,index = [x for x in range(0,X_resampled.shape[0],1)])
train_resample_y = pd.DataFrame({'Label':y_resampled})
train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=500,
 max_depth=7,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.9,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=7,
 scale_pos_weight=1,
 seed=27,
 #objective= 'multi:softmax' #多分类的问题
 )
output = modelfit(xgb1, train_resampled, predictors)

### find out mis classified

tmp2=pd.DataFrame({'ID':np.array(train['ID']),'real_label':np.array(train['Label']),'pred_label':output[1]})

error_cls=tmp2[tmp2['real_label'] != tmp2['pred_label']]

tmp3=pd.merge(error_cls,mix_train,on='ID')


tmp=pd.DataFrame({'ID':output['ID'],'LABEL':output['Label']})
tmp.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/xgSmo_2_pred.csv",index=False,sep=',')


