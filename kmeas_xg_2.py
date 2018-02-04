#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####  把测试集也加入做聚类
#####   重复多次取预测平均


####产出1，1/20 正例
####产出2， 1/30 正例


"""
Created on Wed Jan 10 11:49:08 2018

@author: 黄浩桂
"""
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

#创建要聚类的数据集
data=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")
data_pre=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")


####  把时间变量按照半小时切片
data['V_Time']=data['V_Time']/1800
data[['V_Time']] = data[['V_Time']].astype(int)
data_pre['V_Time']=data_pre['V_Time']/1800
data_pre[['V_Time']] = data_pre[['V_Time']].astype(int)


X=data.iloc[:,1:-1]
X_pred=data_pre.iloc[:,1:]
X_all=pd.concat([X,X_pred])

#根据图来看K选多少，elbow_method
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,#分多少类
                init='k-means++',#Method for initialization:
                n_init=10,#K次划分，默认划分10次
                max_iter=300,#迭代次数
                random_state=0)
    km.fit(X_all)    #用训练集和测试一起来聚类
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
#由图可见 类别数分成3类，即K=3为最佳
#silhouette analysis 轮廓分析，判定K-means模型的质量
#其计算步骤如下：
#对于第 i 个对象，计算它到所属簇中所有其他对象的平均距离，记 ai （体现凝聚度）
#对于第 i 个对象和不包含该对象的任意簇，计算该对象到给定簇中所有对象的平均距离，记 bi （体现分离度）
#第 i 个对象的轮廓系数为 si = (bi-ai)/max(ai, bi)  //回头研究一下 wordpress 的公式插件去
km = KMeans(n_clusters=3,#分多少类
                init='k-means++',#Method for initialization:
                n_init=10,#K次划分，默认划分10次
                max_iter=300,#迭代次数
                tol=1e-04,
                random_state=0)
km.fit(X_all) 


#km = KMeans(n_clusters=3,
#            init='k-means++',
#            n_init=10,
#            max_iter=300,
#            tol=1e-04,
#            random_state=0)
y_km = km.fit_predict(X) #产生预测值


#产生预测集的标签
#km = KMeans(n_clusters=3,
#            init='k-means++',
#            n_init=10,
#            max_iter=300,
#            tol=1e-04,
#            random_state=0)
y_km_pred = km.fit_predict(X_pred) #产生预测值



'''cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
                                     y_km,
                                     metric='euclidean')#算力不够
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()
#将原数据和聚类的值进行结合
'''
data_kmeans=pd.concat([data,pd.DataFrame({'y_km':y_km})],axis=1)
data_kmeans_pred=pd.concat([data_pre,pd.DataFrame({'y_km_pred':y_km_pred})],axis=1)

#按聚类标签分成3组:
cluster_1=data_kmeans.ix[y_km==0,:-1]
cluster_2=data_kmeans.ix[y_km==1,:-1]
cluster_3=data_kmeans.ix[y_km==2,:-1]

#预测集分成3组
cluster_pre1=data_kmeans_pred.ix[y_km_pred==0,:-1]
cluster_pre2=data_kmeans_pred.ix[y_km_pred==1,:-1]
cluster_pre3=data_kmeans_pred.ix[y_km_pred==2,:-1]



#每个聚类完成的数据进行xgb
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import f1_score  ,confusion_matrix
from sklearn import  metrics   #Additional     scklearn functions

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
        dtrain_predictions = alg.predict(dtrain[predictors])#产出smot每个ID的0和1
        pred_y = alg.predict(pred[predictors])#产出预测集的0和1
        ori_pred = alg.predict(val[predictors])#产出原始数据集的0和1
        output = pd.concat([pred,pd.DataFrame({'Label':pred_y})],axis = 1)#合并输出
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]#每个id的预测概率
        
        #Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        print("f1_score:%f" % f1_score(np.array(dtrain[target]),dtrain_predictions))
        print(confusion_matrix(np.array(dtrain[target]),dtrain_predictions))
        print("f1_score_ori:%f" % f1_score(np.array(val[target]),ori_pred))
        print("ori_confuse:")
        print(confusion_matrix(np.array(val[target]),ori_pred))
        print(pd.DataFrame(pred_y)[0].value_counts())
        return [output,ori_pred]

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=500,
 max_depth=7,
 min_child_weight=1,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=7,
 scale_pos_weight=1,
 seed=27)

from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
#第一个群
## smote 比例
l=[200,150,120,80,70,60,50,30,20,10]
train,val = train_test_split(cluster_1, test_size = 0.3,random_state=43)
train_pred_all_c1=val['ID']
test_pred_all_c1=cluster_pre1['ID']


#####
from imblearn.ensemble import BalanceCascade 
bc = BalanceCascade(random_state=42)

for iter in range(10):
    
    train_x = train.iloc[:,1:-1]
    train_y = train['Label']
    #sm = SMOTEENN(ratio={1:2000})##10太小，可以调大
    sm = SMOTE(ratio={1:5000},k_neighbors=2,kind='borderline1')
    X_resampled, y_resampled = sm.fit_sample(np.array(train_x), np.array(train_y))
    #X_resampled, y_resampled = bc.fit_sample(np.array(train_x), np.array(train_y))
    #X_resampled, y_resampled =train_x,train_y
    pred = cluster_pre1
    pred = pred.reset_index()
    pred=pred.drop('index',1)
    
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    #train_resample_x=X_resampled
    train_resample_x = pd.DataFrame(X_resampled,columns = predictors,index = [x for x in range(0,X_resampled.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_resampled})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)
    
    output = modelfit(xgb1, train_resampled, predictors)
    test_pred_all_c1 = np.c_[test_pred_all_c1,output[0]['Label']]
    train_pred_all_c1=np.c_[train_pred_all_c1,output[1]]
    
###  检验集结果
train_pred_c1=train_pred_all_c1[:,1:].sum(axis=1)
train_pred_c1[train_pred_c1<5]=0
train_pred_c1[train_pred_c1>=5]=1
print("f1_score_ori:%f" % f1_score(np.array(val[target]),train_pred_c1))
    
test_pred_c1=test_pred_all_c1[:,1:].sum(axis=1)
test_pred_c1[test_pred_c1<5]=0
test_pred_c1[test_pred_c1>=5]=1
test_out_c1=pd.DataFrame({'ID':cluster_pre1['ID'],'Label':test_pred_c1})



test_out_c1=pd.DataFrame({'ID':np.array(cluster_pre1['ID']),'Label':np.array(a)})

ttt2=test_out_c1
#第二个群
from imblearn.over_sampling import ADASYN 
from imblearn.under_sampling import TomekLinks 
from imblearn.ensemble import EasyEnsemble 
ee = EasyEnsemble(ratio={0:500})
enn = TomekLinks()

train,val = train_test_split(cluster_2, test_size = 0.3,random_state=1)
train_pred_all_c2=val['ID']
test_pred_all_c2=cluster_pre2['ID']

for iter in range(10):
    train_x = train.iloc[:,1:-1]
    train_y = train['Label']
    #sm = SMOTEENN(ratio={1:int(train.shape[0]/l[iter])})
    #sm = SMOTE(ratio={1:500},k_neighbors=7,kind='borderline1')
    #ada = ADASYN(ratio={1:200},n_neighbors=10)
    X_resampled, y_resampled = enn.fit_sample(np.array(train_x), np.array(train_y))
    #X_resampled, y_resampled = sm.fit_sample(np.array(train_x), np.array(train_y))
    #X_resampled, y_resampled =train_x,train_y
    pred = cluster_pre2
    pred = pred.reset_index()
    pred=pred.drop('index',1)
 

    predictors = [x for x in train.columns if x not in [target,IDcol]]
    #train_resample_x=X_resampled
    train_resample_x = pd.DataFrame(X_resampled,columns = predictors,index = [x for x in range(0,X_resampled.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_resampled})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)
    
    output = modelfit(xgb1, train_resampled, predictors)
    test_pred_all_c2 = np.c_[test_pred_all_c2,output[0]['Label']]
    train_pred_all_c2=np.c_[train_pred_all_c2,output[1]]
    
###  检验集结果
train_pred_c2=train_pred_all_c2[:,1:].sum(axis=1)
train_pred_c2[train_pred_c2<5]=0
train_pred_c2[train_pred_c2>=5]=1
print("f1_score_ori:%f" % f1_score(np.array(val[target]),train_pred_c2))
        
test_pred_c2=test_pred_all_c2[:,1:].sum(axis=1)
test_pred_c2[test_pred_c2<5]=0
test_pred_c2[test_pred_c2>=5]=1
test_out_c2=pd.DataFrame({'ID':cluster_pre2['ID'],'Label':test_pred_c2})



#第三个群
train,val = train_test_split(cluster_3, test_size = 0.3,random_state=1)

train_pred_all_c3=val['ID']
test_pred_all_c3=cluster_pre3['ID']


for iter in range(10):
    
    train_x = train.iloc[:,1:-1]
    train_y = train['Label']
   # sm = SMOTEENN(ratio={1:int(train.shape[0]/l[iter])})
    sm = SMOTE(ratio={1:6000},k_neighbors=2,kind='borderline2')
    X_resampled, y_resampled = sm.fit_sample(np.array(train_x), np.array(train_y))
    #X_resampled, y_resampled =train_x,train_y
    pred = cluster_pre3
    pred = pred.reset_index()
    pred=pred.drop('index',1)
   
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    #train_resample_x=X_resampled
    train_resample_x = pd.DataFrame(X_resampled,columns = predictors,index = [x for x in range(0,X_resampled.shape[0],1)])
    train_resample_y = pd.DataFrame({'Label':y_resampled})
    train_resampled = pd.concat([train_resample_x,train_resample_y],axis=1)

    output = modelfit(xgb1, train_resampled, predictors)
    test_pred_all_c3 = np.c_[test_pred_all_c3,output[0]['Label']]
    train_pred_all_c3=np.c_[train_pred_all_c3,output[1]]
    
 ###  检验集结果
train_pred_c3=train_pred_all_c3[:,1:].sum(axis=1)
train_pred_c3[train_pred_c3<5]=0
train_pred_c3[train_pred_c3>=5]=1
print("f1_score_ori:%f" % f1_score(np.array(val[target]),train_pred_c3))   
    
    
    
test_pred_c3=test_pred_all_c3[:,1:].sum(axis=1)
test_pred_c3[test_pred_c3<5]=0
test_pred_c3[test_pred_c3>=5]=1
test_out_c3=pd.DataFrame({'ID':cluster_pre3['ID'],'Label':test_pred_c3})


ttttt=test_out_c3
#合并三类数据集
k_means_final=pd.concat([test_out_c1,test_out_c2,test_out_c3])
k_means_final[['Label']] = k_means_final[['Label']].astype(int)
k_means_final['Label'].value_counts()

k_means_final.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/kmeans_xg_3.csv",index=False,sep=',')



#k_means_final.to_csv("D:/k_means_final.csv",index=False,columns=['ID','label'],sep=',')










