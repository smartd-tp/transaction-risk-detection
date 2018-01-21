#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:04:03 2018

@author: cp
"""

import pandas            as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


train=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")

test_ori=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")

sm = SMOTE(k_neighbors=2,kind='regular')
#sm = SMOTEENN()
x=train.iloc[:,2:-1]
y=train.loc[:,'Label']
x_test=test_ori.iloc[:,2:]

###  5个不同正例比例的训练集
#sm1 = SMOTE(ratio={1:15000},k_neighbors=2,kind='regular')
#sm2 = SMOTE(ratio={1:7000},k_neighbors=2,kind='regular')
#sm3 = SMOTE(ratio={1:2000},k_neighbors=2,kind='regular')
#sm4 = SMOTE(ratio={1:1000},k_neighbors=2,kind='regular')
#sm5 = SMOTE(ratio={1:600},k_neighbors=2,kind='regular')

sme1 = SMOTEENN(ratio={1:15000},random_state=10,smote=sm1)
sme2 = SMOTEENN(ratio={1:7000},random_state=10,smote=sm2)
sme3 = SMOTEENN(ratio={1:2000},random_state=10,smote=sm3)
sme4 = SMOTEENN(ratio={1:1000},random_state=10,smote=sm4)
sme5 = SMOTEENN(ratio={1:600},random_state=10,smote=sm5)


X_resampled_1, y_resampled_1 = sme1.fit_sample(x,y)
X_resampled_2, y_resampled_2 = sme2.fit_sample(x,y)
X_resampled_3, y_resampled_3 = sme3.fit_sample(x,y)
X_resampled_4, y_resampled_4 = sme4.fit_sample(x,y)
X_resampled_5, y_resampled_5 = sme5.fit_sample(x,y)


#####    balancecascade






train_1=pd.DataFrame(np.c_[X_resampled_1,y_resampled_1])
train_2=pd.DataFrame(np.c_[X_resampled_2,y_resampled_2])
train_3=pd.DataFrame(np.c_[X_resampled_3,y_resampled_3])
train_4=pd.DataFrame(np.c_[X_resampled_4,y_resampled_4])
train_5=pd.DataFrame(np.c_[X_resampled_5,y_resampled_5])
#train_5 = pd.DataFrame(np.c_[x,y])

train_1.columns=list(train.columns.values)[2:]
train_2.columns=list(train.columns.values)[2:]
train_3.columns=list(train.columns.values)[2:]
train_4.columns=list(train.columns.values)[2:]
train_5.columns=list(train.columns.values)[2:]


xgb_train_1 = xgb.DMatrix(train_1.drop(['Label'],axis=1), train_1['Label'])
xgb_train_2 = xgb.DMatrix(train_2.drop(['Label'],axis=1), train_2['Label'])
xgb_train_3 = xgb.DMatrix(train_3.drop(['Label'],axis=1), train_3['Label'])
xgb_train_4 = xgb.DMatrix(train_4.drop(['Label'],axis=1), train_4['Label'])
xgb_train_5 = xgb.DMatrix(train_5.drop(['Label'],axis=1), train_5['Label'])



###  测试集： 原训练集
val_y=train['Label']
val_X=train.iloc[:,2:-1]

###  测试集：
test_x=test_ori.iloc[:,2:]


### 验证集
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_test=xgb.DMatrix(test_x)
watchlist1 = [(xgb_train_1, 'train'),(xgb_val, 'test')]  
watchlist2 = [(xgb_train_2, 'train'),(xgb_val, 'test')]  
watchlist3 = [(xgb_train_3, 'train'),(xgb_val, 'test')]  
watchlist4 = [(xgb_train_4, 'train'),(xgb_val, 'test')]  
watchlist5 = [(xgb_train_5, 'train'),(xgb_val, 'test')]  
###  训练5个xgboost分类器


#输入参数
params={
'booster':'gbtree',#运用提升树模型
'objective': 'multi:softmax', #多分类的问题
'num_class':2, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':7, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.9, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':1, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.01, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}


####  xg classifier
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

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
target = 'Label'
IDcol = 'ID'
def modelfit(alg, dtrain, predictors,testdata,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
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
        pred_y = alg.predict(testdata[predictors])
        ori_pred = alg.predict(train[predictors])
        output = pd.concat([testdata,pd.DataFrame({'Label':pred_y})],axis = 1)
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

predictors = [x for x in train.columns if x not in [target,IDcol,'V_Time']]


output1 = modelfit(xgb1, train_1, predictors,test_ori)
output2 = modelfit(xgb1, train_2, predictors,test_ori)
output3 = modelfit(xgb1, train_3, predictors,test_ori)
output4 = modelfit(xgb1, train_4, predictors,test_ori)
output5 = modelfit(xgb1, train_5, predictors,test_ori)

ypred_1=np.array(output1[0]['Label'])
ypred_2=np.array(output2[0]['Label'])
ypred_3=np.array(output3[0]['Label'])
ypred_4=np.array(output4[0]['Label'])
ypred_5=np.array(output5[0]['Label'])

ypred1=output1[1]
ypred2=output2[1]
ypred3=output3[1]
ypred4=output4[1]
ypred5=output5[1]




#生成xgboost模型
bst1=xgb.train(params,xgb_train_1,num_boost_round=30,evals=watchlist1)
bst2=xgb.train(params,xgb_train_2,num_boost_round=30,evals=watchlist2)
bst3=xgb.train(params,xgb_train_3,num_boost_round=30,evals=watchlist3)
bst4=xgb.train(params,xgb_train_4,num_boost_round=30,evals=watchlist4)
bst5=xgb.train(params,xgb_train_5,num_boost_round=30,evals=watchlist5)

## 随机森林模型
modelRF = RandomForestClassifier(n_estimators=500,max_depth=5, random_state=0)
modelRF.fit(x, y)
#ypred6=modelRF.predict(x)



##  预测
ypred1=bst1.predict(xgb_val)
ypred2=bst2.predict(xgb_val)
ypred3=bst3.predict(xgb_val)
ypred4=bst4.predict(xgb_val)
ypred5=bst5.predict(xgb_val)
ypred6=modelRF.predict(x)


##  预测test
ypred_1=bst1.predict(xgb_test)
ypred_2=bst2.predict(xgb_test)
ypred_3=bst3.predict(xgb_test)
ypred_4=bst4.predict(xgb_test)
ypred_5=bst5.predict(xgb_test)
ypred_6=modelRF.predict(x_test)





####  筛选预测不一致的点
filter_result=pd.DataFrame(data={'ID':train['ID'],'c1':ypred1,'c2':ypred2,'c3':ypred3,'c4':ypred4,'c5':ypred5,'c6':ypred6,'Label':train['Label']})


filter_result['vote']=filter_result.iloc[:,2:8].sum(axis=1)

mix_train=filter_result[(filter_result['vote']>0) & (filter_result['vote']<6)]

filter_result2=pd.DataFrame(data={'ID':test_ori['ID'],'c1':ypred_1,'c2':ypred_2,'c3':ypred_3,'c4':ypred_4,'c5':ypred_5,'c6':ypred_6})

filter_result2['vote']=filter_result2.iloc[:,1:7].sum(axis=1)

mix_test=filter_result2[(filter_result2['vote']>0) & (filter_result2['vote']<6)]

###  比较易错点的相对距离

mix_train_data = pd.merge(train.drop(['V_Time'],axis=1),pd.DataFrame(mix_train['ID']))

mix_test_data = pd.merge(test_ori.drop(['V_Time'],axis=1),pd.DataFrame(mix_test['ID']))


####   距离计算
def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    return ED

dis=EuclideanDistances(mix_test_data.drop(['ID'],axis=1), mix_train_data.drop(['ID','Label'],axis=1)  )
###  寻照最近点
l=mix_test_data.shape[0]
closest_pt=np.linspace(0,1,l)

for i  in range(l):
    a=dis[i,].tolist()
    closest_pt[i]=a[0].index(np.min(a))


##  赋给最近点的label
    
mix_new_pred=np.linspace(0,1,l)
for i in range(l): 
    mix_new_pred[i]=mix_train_data['Label'][int(closest_pt[i])]
    

sum(mix_new_pred)


mix.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/mix_train_pts.csv",index=False,sep=',')

mix


c=filter_result2[(filter_result2['vote']==6) 

d=filter_result[(filter_result['vote']==6 ) & (filter_result['Label']==1)]

sum(mix_train['Label'])


print(mix['c1'].sum())
print(mix['c2'].sum())
print(mix['c3'].sum())
print(mix['c4'].sum())
print(mix['c5'].sum())
print(mix['Label'].sum())
###  比较易错点的预测情况





#####    结合两部份预测

tmp1=pd.DataFrame({'ID':mix_test['ID'],'Label':mix_new_pred})
pred_p1=tmp1[tmp1['Label']==1]
pred_p2=pd.DataFrame({'ID':c['ID'],'Label':c['c1']})

pred_final=pd.DataFrame({'ID':test_ori['ID'],'Label':0})

for i in range(100000):
    if pred_final.loc[i,'ID'] in list(pred_p1['ID']) :
        pred_final.loc[i,'Label']=1


for i in range(100000):
    if pred_final.loc[i,'ID'] in list(pred_p2['ID']) :
        pred_final.loc[i,'Label']=1
        
      

pred_final.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/mixcls_xg_enn_rf.csv",index=False,sep=',')


#np.sqrt(np.sum(np.square(mix_train_data.iloc[61,1:] - mix_test_data.iloc[1,1:])))  




##############  method 2 : 把易错的点单独做分类： rf， 逻辑回归， knn

## 随机森林模型
modelRF = RandomForestClassifier(n_estimators=500,max_depth=20, random_state=0)
modelRF.fit(mix_train_data.iloc[:,1:-1],mix_train_data['Label'] )
mix_new_pred_rf=modelRF.predict(mix_test_data.iloc[:,1:])
sum(mix_new_pred_rf)

###逻辑回归
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(C=0.1, tol=0.01,penalty='l2',random_state=0)
clf_LR.fit(mix_train_data.iloc[:,1:-1],mix_train_data['Label'])
mix_new_pred_LR=clf_LR.predict(mix_test_data.iloc[:,1:])
sum(mix_new_pred_LR)


tmp1=pd.DataFrame({'ID':mix_test['ID'],'Label':mix_new_pred_rf})
#####  回到  结合两部份预测 执行


