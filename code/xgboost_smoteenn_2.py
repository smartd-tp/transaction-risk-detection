#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:29:04 2018

@author: cp

output: xgSmo_4_pred /xgSmo_3_pred
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import f1_score  ,confusion_matrix
from sklearn import  metrics   #Additional     scklearn functions
from imblearn.under_sampling import EditedNearestNeighbours 
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split
from matplotlib.pylab import rcParams
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics        
        


train_raw = pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")
pred_raw = pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")

############  参数: 
time_intervals =60    #   时间切片的个粒度  /60 600 1800 3600
ratio = 500          #   smoteenn 的反例个数
xgb1 = XGBClassifier(   #  xboost
 learning_rate =0.1,
 n_estimators=500,
 max_depth=20,
 min_child_weight=1,
 gamma=0,
 subsample=0.85,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=7,
 scale_pos_weight=1,
 reg_lambda=0.1,
 reg_alpha=1e-05,
 seed=27
 )


rcParams['figure.figsize'] = 12, 4
target = 'Label'
IDcol = 'ID'




def modelfit(alg, dtrain, predictors,validation,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
        #Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
        #Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors]) ### 训练集
        pred_y = alg.predict(pred[predictors]) ### 测试集
        val_pred = alg.predict(validation[predictors]) ### 验证集
        output = pd.concat([pred,pd.DataFrame({'Label':pred_y})],axis = 1)
        #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
        #Print model report:
        print("\nModel Report")
       # print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
       # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        print("train_f1_score:%f" % f1_score(np.array(dtrain[target]),dtrain_predictions))
        print(confusion_matrix(np.array(dtrain[target]),dtrain_predictions))
        f1score=f1_score(np.array(validation[target]),val_pred)
        print("f1_score_val:%f" % f1score )
        print("val_confuse:")
        print(confusion_matrix(np.array(validation[target]),val_pred))
        print(pd.DataFrame(pred_y)[0].value_counts())
        return [output,val_pred,f1score]


####   样本参数
sm = SMOTE(ratio= {1:ratio},k_neighbors=2,random_state=42)
enn = EditedNearestNeighbours(kind_sel='mode',n_neighbors=2,random_state=42) ##  kind_sel='all'会更严格， neighbor数量多更严格
smenn = SMOTEENN(sm,enn,n_jobs=4)

########################################

####  把时间变量切片
train=train_raw
pred = pred_raw
train['V_Time']=train['V_Time']/time_intervals
train[['V_Time']] = train[['V_Time']].astype(int)
pred['V_Time']=pred['V_Time']/time_intervals
pred[['V_Time']] = pred[['V_Time']].astype(int)




####抽样，分训练集和验证集
       
train_sample, val=  train_test_split(train, test_size = 0.3,random_state=1)

#### 训练集做smote处理
X_train_sample = train_sample.iloc[:,1:-1]
y_train_sample = train_sample['Label']
X_train_smo,y_train_smo = sm.fit_sample(np.array(X_train_sample), np.array(y_train_sample))


### 网格搜索
tuned_parameters = [{'reg_alpha':[1e-5]}]

clf = GridSearchCV(XGBClassifier(silent=0,nthread=7,learning_rate= 0.1,min_child_weight=1, max_depth=7,gamma=0,subsample=0.85,
                                 colsample_bytree=0.8,reg_lambda=0.1,seed=1000,reg_alpha=1e-05), 
                                   param_grid=tuned_parameters,scoring='f1',n_jobs=4,iid=False,cv=5)  
clf.fit(X_train_smo, y_train_smo)
##clf.grid_scores_, clf.best_params_, clf.best_score_
print(clf.best_params_)
clf.grid_scores_



####   在验证集上测试泛化能力
y_val, y_pred = val['Label'], clf.predict(np.array(val.iloc[:,1:-1]))
print("f1_score_ori:%f" % f1_score(np.array(y_val),y_pred))






###################### 在训练好模型之后，再调整时间切片
time_intervals=[60,600,1800,3600]
for iter in range(4):
    
####  把时间变量切片
    train=train_raw
    pred = pred_raw
    train['V_Time']=train['V_Time']/time_intervals[iter]
    train[['V_Time']] = train[['V_Time']].astype(int)
    pred['V_Time']=pred['V_Time']/time_intervals[iter]
    pred[['V_Time']] = pred[['V_Time']].astype(int)
    
         
    train_sample, val=  train_test_split(train, test_size = 0.3,random_state=1)

#### 训练集做smote处理
    X_train_sample = train_sample.iloc[:,1:-1]
    y_train_sample = train_sample['Label']
    X_train_smo,y_train_smo = sm.fit_sample(np.array(X_train_sample), np.array(y_train_sample))
    
    

###  训练
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    train_resample_x = pd.DataFrame(X_train_smo,columns = predictors,index = [x for x in range(0,X_train_smo.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_train_smo})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

    output = modelfit(xgb1, train_resampled, predictors,val)


#######################  再调整采样模型
#rt =[500,1500,2000,5000,8000,10000]    500最优   
watchlist =[[0 for i in range(5)] for j in range(6)]  
for iter in range(6):
      
####  把时间变量切片
    train=train_raw
    pred = pred_raw
    train['V_Time']=train['V_Time']/60
    train[['V_Time']] = train[['V_Time']].astype(int)
    pred['V_Time']=pred['V_Time']/60
    pred[['V_Time']] = pred[['V_Time']].astype(int)
    
         
    train_sample, val=  train_test_split(train, test_size = 0.3,random_state=1)

######   smote 比例
    sm = SMOTE(ratio= {1:500},k_neighbors=2)
    enn = EditedNearestNeighbours(kind_sel='all',n_neighbors=2) ##  kind_sel='all'会更严格， neighbor数量多更严格
    smenn = SMOTEENN(sm,enn)
    
######    每一个smote的比例重复5次削减随机性
    for i in range(3):
        #### 训练集做smote处理
        X_train_sample = train_sample.iloc[:,1:-1]
        y_train_sample = train_sample['Label']
        X_train_smo,y_train_smo = sm.fit_sample(np.array(X_train_sample), np.array(y_train_sample))
    
    

###  训练
        predictors = [x for x in train.columns if x not in [target,IDcol]]
        train_resample_x = pd.DataFrame(X_train_smo,columns = predictors,index = [x for x in range(0,X_train_smo.shape[0],1)])
        train_resample_y = pd.DataFrame({'Label':y_train_smo})
        train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

        output = modelfit(xgb1, train_resampled, predictors,val)
        watchlist[1][i]=output[2]

        



######   抽样10次，建模取均值
test_pred_record=pred['ID']
for iter in range(10):
     train_sample, val=  train_test_split(train, test_size = 0.3)
     sm = SMOTE(ratio= {1:2000},k_neighbors=2)
     enn = EditedNearestNeighbours(kind_sel='all',n_neighbors=2) ##  kind_sel='all'会更严格， neighbor数量多更严格
     smenn = SMOTEENN(sm,enn)
     X_train_sample = train_sample.iloc[:,1:-1]
     y_train_sample = train_sample['Label']
     X_train_smo,y_train_smo = sm.fit_sample(np.array(X_train_sample), np.array(y_train_sample))
     predictors = [x for x in train.columns if x not in [target,IDcol]]
     train_resample_x = pd.DataFrame(X_train_smo,columns = predictors,index = [x for x in range(0,X_train_smo.shape[0],1)])
     train_resample_y = pd.DataFrame({'Label':y_train_smo})
     train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

     output = modelfit(xgb1, train_resampled, predictors,val)
     test_pred_record = np.c_[test_pred_record,output[0]['Label']]
     
     
test_output = test_pred_record[:,1:].sum(axis=1)
pd.DataFrame(test_output)[0].value_counts()
test_output[test_output<5]=0
test_output[test_output>=5]=1

out=pd.DataFrame({'ID':pred['ID'],'Label':test_output})

out.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/xgSmo_5_pred.csv",index=False,sep=',')






























       
    

