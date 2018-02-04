#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:40:56 2018

@author: cp
"""

#####xg_stack_3    xgstacking + 3 模型，  剔除100%预测命中
#####xg_stack_4    LRstacking + 3 模型，  剔除100%预测命中
######xg_stack_5    xgstacking + 3 模型， 加入了一般测试集
######xg_stack_6    xgstacking + 3 模型， 平均投票结果

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression      
        

########## 通过10个不同比例的smote模型训练xgboost分类器，然后用10个模型的的结果作为下一阶段xgboost输入


train_raw = pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")
pred_raw = pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")

############  参数: 
time_intervals =60    #   时间切片的个粒度  /60 600 1800 3600
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


def xgbfit(alg, dtrain, predictors,prediction,useTrainCV=True, cv_folds=2, early_stopping_rounds=50):
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
        pred_y = alg.predict(prediction[predictors]) ### 测试集
        #val_pred = alg.predict(validation[predictors]) ### 验证集
        #output = pd.concat([prediction,pd.DataFrame({'Label':pred_y})],axis = 1)
        #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
        #Print model report:
        print("\nModel Report")
       # print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
       # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        print("train_f1_score:%f" % f1_score(np.array(dtrain[target]),dtrain_predictions))
        print(confusion_matrix(np.array(dtrain[target]),dtrain_predictions))
    #    f1score=f1_score(np.array(validation[target]),val_pred)
       # print("f1_score_val:%f" % f1score )
       # print("val_confuse:")
       # print(confusion_matrix(np.array(validation[target]),val_pred))
        print(pd.DataFrame(pred_y)[0].value_counts())
        return pred_y



    
####  把时间变量切片
train=train_raw
pred = pred_raw
train['V_Time']=train['V_Time']/time_intervals
train[['V_Time']] = train[['V_Time']].astype(int)
pred['V_Time']=pred['V_Time']/time_intervals
pred[['V_Time']] = pred[['V_Time']].astype(int)
    
 #####  作为stacking 模型的训练和验证集        
train_sample, val=  train_test_split(train, test_size = 0.3)
predictors = [x for x in train.columns if x not in [target,IDcol]]

####  stacking输入数据
stacking_input=train_sample['ID']

##########  11个xg模型字典 
model_list={1:'xgb1',2:'xgb2',3:'xgb3',4:'xgb4',5:'xgb5',6:'xgb6',7:'xgb7',8:'xgb8',9:'xgb9',10:'xgb10'}
for i in range(11):
    model_list[i] = XGBClassifier(   #  xboost
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



smote_ratio=[0,500,2000,3000,4000,5000,6000,7000,8000,9000,10000]


for iter in range(11):
    if iter==0 :
       X_train_smo,y_train_smo = train.iloc[:,1:-1],train['Label'] 
    else :
       sm = SMOTE(ratio= {1:smote_ratio[iter]},k_neighbors=2)
       enn = EditedNearestNeighbours(kind_sel='all',n_neighbors=2) ##  kind_sel='all'会更严格， neighbor数量多更严格
       smenn = SMOTEENN(sm,enn) 
       X_train_smo,y_train_smo = sm.fit_sample(np.array(train.iloc[:,1:-1]), np.array(train['Label'] ))
    
    train_resample_x = pd.DataFrame(X_train_smo,columns = predictors,index = [x for x in range(0,X_train_smo.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_train_smo})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

    output = xgbfit(model_list[iter], train_resampled, predictors,train_sample)
    stacking_input=np.c_[stacking_input,output]    
        



########  逻辑回归做为stacking input
stacking_lr_input=train_sample['ID']
val_lr_set=val['ID'] 
test_lr_set= pred['ID']
##########  11个xg模型字典 
model_lr_list={1:'xgb1',2:'xgb2',3:'xgb3',4:'xgb4',5:'xgb5',6:'xgb6',7:'xgb7',8:'xgb8',9:'xgb9',10:'xgb10'}
for i in range(11):
    model_lr_list[i] = LogisticRegression(penalty='l2', tol=1e-5,C=0.1,random_state=0,max_iter=100)
            

smote_ratio=[0,500,2000,3000,4000,5000,6000,7000,8000,9000,10000]


for iter in range(11):
    if iter==0 :
       X_train_smo,y_train_smo = train.iloc[:,1:-1],train['Label'] 
    else :
       sm = SMOTE(ratio= {1:smote_ratio[iter]},k_neighbors=2)
       enn = EditedNearestNeighbours(kind_sel='all',n_neighbors=2) ##  kind_sel='all'会更严格， neighbor数量多更严格
       smenn = SMOTEENN(sm,enn) 
       X_train_smo,y_train_smo = sm.fit_sample(np.array(train.iloc[:,1:-1]), np.array(train['Label'] ))
    
    train_resample_x = pd.DataFrame(X_train_smo,columns = predictors,index = [x for x in range(0,X_train_smo.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_train_smo})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

    model_lr_list[iter].fit(train_resampled[predictors],train_resampled[target])   
    output =  model_lr_list[iter].predict(train_sample[predictors])
    stacking_lr_input=np.c_[stacking_lr_input,output]       #######  stacking input
    val_x = model_lr_list[iter].predict(val[predictors]) 
    test_x = model_lr_list[iter].predict(pred[predictors])
    val_lr_set=np.c_[val_lr_set,val_x]            #####  validation input
    test_lr_set=np.c_[test_lr_set,test_x]     #####  test input
    


##########  随机森林作为stacking input
stacking_rf_input=train_sample['ID']
val_rf_set=val['ID'] 
test_rf_set= pred['ID']
##########  11个xg模型字典 
model_rf_list={1:'xgb1',2:'xgb2',3:'xgb3',4:'xgb4',5:'xgb5',6:'xgb6',7:'xgb7',8:'xgb8',9:'xgb9',10:'xgb10'}
for i in range(11):
    model_rf_list[i] = RandomForestClassifier(random_state = 0, n_jobs = 7,min_samples_leaf=1 ,criterion='entropy',
                            max_features= 0.8,min_impurity_decrease= 1e-5,n_estimators =300 , max_depth = 11)
            
smote_ratio=[0,500,2000,3000,4000,5000,6000,7000,8000,9000,10000]


for iter in range(11):
    if iter==0 :
       X_train_smo,y_train_smo = train.iloc[:,1:-1],train['Label'] 
    else :
       sm = SMOTE(ratio= {1:smote_ratio[iter]},k_neighbors=2)
       enn = EditedNearestNeighbours(kind_sel='all',n_neighbors=2) ##  kind_sel='all'会更严格， neighbor数量多更严格
       smenn = SMOTEENN(sm,enn) 
       X_train_smo,y_train_smo = sm.fit_sample(np.array(train.iloc[:,1:-1]), np.array(train['Label'] ))
    
    train_resample_x = pd.DataFrame(X_train_smo,columns = predictors,index = [x for x in range(0,X_train_smo.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_train_smo})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

    model_rf_list[iter].fit(train_resampled[predictors],train_resampled[target])   
    output =  model_rf_list[iter].predict(train_sample[predictors])
    stacking_rf_input=np.c_[stacking_rf_input,output]       #######  stacking input
    val_x = model_rf_list[iter].predict(val[predictors]) 
    test_x = model_rf_list[iter].predict(pred[predictors])
    val_rf_set=np.c_[val_rf_set,val_x]            #####  validation input
    test_rf_set=np.c_[test_rf_set,test_x]     #####  test input
    









###########################    最终版：用原始训练集通过30组模型，生成stacking的输入，同理产生测试集
temp1=np.c_[stacking_input[:,2:],stacking_lr_input[:,1:],stacking_rf_input[:,1:]]
temp2= np.c_[val_set[:,2:],val_lr_set[:,1:],val_rf_set[:,1:]]   
final_stacking_put = np.r_[temp1,temp2]    
temp1=train_sample['Label']    
temp2=val['Label']
final_stacking_y=np.r_[temp1,temp2]  
final_test_input=np.c_[test_set[:,2:],test_lr_set[:,1:],test_rf_set[:,1:]]

pd.DataFrame(final_stacking_put).to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/stacking_final_input.csv",index=False,sep=',')
pd.DataFrame(final_stacking_y).to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/final_stacking_y.csv",index=False,sep=',')
pd.DataFrame(final_test_input).to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/final_test_input.csv",index=False,sep=',')



#####################将测试集作为第二层的输入,投票结果作为假设的label

final_stacking_put=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/stacking_final_input.csv")
final_stacking_y=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/final_stacking_y.csv")
final_test_input=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/final_test_input.csv")

final_stacking_put=np.array(final_stacking_put)
final_stacking_y=np.array(final_stacking_y)
final_test_input=np.array(final_test_input)


test_y_hp=final_test_input.sum(axis=1)
test_y_hp[test_y_hp<=15]=0
test_y_hp[test_y_hp>15]=1

new_test_train = np.c_[final_test_input,test_y_hp]
new_test_train2,delete = train_test_split(new_test_train, test_size = 0.7,random_state=1)

new_stacking_input=np.r_[final_stacking_put,new_test_train2[:,:-1]]
new_stacking_y=np.r_[final_stacking_y[:,0],new_test_train2[:,-1]]




#####  stacking validation set
val_set=val['ID']    
for j in range(11):
    val_x = model_list[j].predict(val[predictors]) 
    val_set=np.c_[val_set,val_x]  
    
##### stacking testing set
test_set=   pred['ID']
for j in range(11):
    test_x = model_list[j].predict(pred[predictors]) 
    test_set=np.c_[test_set,test_x]  
     




xgb_stack = XGBClassifier(   #  xboost
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
xgb_param = xgb_stack.get_xgb_params()
xgtrain = xgb.DMatrix(final_stacking_put, label=final_stacking_y)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_stack.get_params()['n_estimators'], nfold=2,
            metrics='auc', early_stopping_rounds=50)
xgb_stack.set_params(n_estimators=cvresult.shape[0])
xgb_stack.fit(new_stacking_input, new_stacking_y,eval_metric='auc')
        
#Predict training set:
dtrain_predictions = xgb_stack.predict(new_stacking_input) ### 训练集
pred_y = xgb_stack.predict(final_test_input) ### 测试集
#val_pred = xgb_stack.predict(np.c_[val_set[:,2:],val_lr_set[:,1:],val_rf_set[:,1:]]) ### 验证集
        #output = pd.concat([prediction,pd.DataFrame({'Label':pred_y})],axis = 1)
        #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
        #Print model report:
print("\nModel Report")
       # print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
       # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
print("train_f1_score:%f" % f1_score(new_stacking_y,dtrain_predictions))
print(confusion_matrix(new_stacking_y,dtrain_predictions))
#f1score=f1_score(np.array(val['Label']),val_pred)
#print("f1_score_val:%f" % f1score )
print("val_confuse:")
#print(confusion_matrix(np.array(val['Label']),val_pred))
print(pd.DataFrame(pred_y)[0].value_counts())    
        

#######洛基回归stacking

stack_lr=LogisticRegression(penalty='l2', tol=1e-5,C=0.1,random_state=0,max_iter=300)
            
stack_lr.fit(np.c_[stacking_input[:,2:],stacking_lr_input[:,1:],stacking_rf_input[:,1:]], train_sample['Label'])
val_pred=stack_lr.predict(np.c_[val_set[:,2:],val_lr_set[:,1:],val_rf_set[:,1:]])
f1score=f1_score(np.array(val['Label']),val_pred)
print("f1_score_val:%f" % f1score )
print("val_confuse:")
print(confusion_matrix(np.array(val['Label']),val_pred))
lr_pred_train=stack_lr.predict(np.c_[test_set[:,2:],test_lr_set[:,1:],test_rf_set[:,1:]])

lr_pred_train.sum()


#######随机森林

from sklearn.ensemble import RandomForestClassifier

train_feature= stacking_input[:,1:]
train_tag=train_sample['Label']
modelRF = RandomForestClassifier(max_depth=11, random_state=0)
modelRF.fit(train_feature, train_tag)
val_pred=modelRF.predict(val_set[:,1:])
f1score=f1_score(np.array(val['Label']),val_pred)
print("f1_score_val:%f" % f1score )
print("val_confuse:")
print(confusion_matrix(np.array(val['Label']),val_pred))
RF_pred_train=modelRF.predict(test_set[:,1:])

RF_pred_train.sum()


out=pd.DataFrame({'ID':pred['ID'],'Label':ensemble})

out.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/xg_stack_esm.csv",index=False,sep=',')







            
        