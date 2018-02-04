#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:43:51 2017

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


train=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")



sm = SMOTE(ratio='minority',k_neighbors=2,kind='regular')
#sm = SMOTEENN()

x=train.iloc[:,2:-1]
y=train.loc[:,'Label']
X_resampled, y_resampled = sm.fit_sample(x,y)


train_smo = pd.DataFrame(np.c_[X_resampled,y_resampled])
train_smo.columns=list(train.columns.values)[2:]


train_norm=train_smo[train_smo["Label"]==0]
train_outlier=train_smo[train_smo["Label"]==1]

outlier_sam=train_outlier.sample(5000)

new_train=pd.DataFrame(np.r_[train_norm,outlier_sam])
new_train.columns=list(train.columns.values)[2:]


test_ori=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")


train_xy,val = train_test_split(new_train, test_size = 0.3,random_state=1)
y = train_xy['Label']
X = train_xy.drop(['Label'],axis=1)

######new test
val_y=train['Label']
val_X=train.iloc[:,2:-1]

#val_y = val['Label']
#val_X = val.drop(['Label'],axis=1)




#输入参数
params={
'booster':'gbtree',#运用提升树模型
'objective': 'multi:softmax', #多分类的问题
'num_class':2, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':13, # 构建树的深度，越大越容易过拟合
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

plst = list(params.items())
num_rounds = 5000 # 迭代次数
#将数据做成矩阵形式
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
watchlist = [(xgb_train, 'train'),(xgb_val, 'test')]  
xgb_test = xgb.DMatrix(test_ori.iloc[:,2:])


#生成xgboost模型
bst=xgb.train(params,xgb_train,num_boost_round=30,evals=watchlist)

#模型指标产出

#将测试集导入至模型中进行预测
ypred=bst.predict(xgb_val)

#ypred=bst.predict(xgb_test)
#y_pred=ypred[(ypred >= 0.5)]

# 设置阈值, 输出一些评价指标，找出1的数量

y_pred =(ypred >= 0.5)

#产出指标和混淆矩阵
from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(val_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(val_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(val_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(val_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(val_y,y_pred))
cmatrix=metrics.confusion_matrix(val_y,y_pred)
print(cmatrix)

#输出观测值打分项
#最后一列是bias
ypred = bst.predict(xgb_val)
print(ypred)

#每个特征的贡献分值

xgb.plot_importance(bst)


ypred_contribs= bst.predict(xgb_val, pred_contribs=True)
score_a = sum(ypred_contribs[0])
print (score_a)
score_b = sum(ypred_contribs[1])
print (score_b)




### output csv
output=pd.DataFrame({'ID':np.array(test_ori.iloc[:,0]),'LABEL':ypred})

output.to_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/xgSmo_1_pred.csv",index=False,sep=',')





### 检视测试结果在变量上的分布
y_pred.dtype='int8'
output=pd.DataFrame({'ID':np.array(train.iloc[:,0]),'LABEL_pred':y_pred+0})
train_all = pd.merge(train,output)

### 0-0
train_00=train_all[(train_all['Label']==0 )& (train_all['LABEL_pred']==0)]
### 0-1
train_01=train_all[(train_all['Label']==0 )& (train_all['LABEL_pred']==1)]
### 1-1
train_11=train_all[(train_all['Label']==1)& (train_all['LABEL_pred']==1)]
### 1-0
train_10=train_all[(train_all['Label']==1)& (train_all['LABEL_pred']==0)]



train_00['tag']=1
train_01['tag']=2
train_11['tag']=3
train_10['tag']=4

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

t1=plt.scatter(train_10[['V_Time']],train_10[['tag']])

t2=plt.scatter(train_01[['V_Time']],train_01[['tag']])
ax.add_patch(t1)
ax.add_patch(t2)

#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)

for i in range(1,33):
    ig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    f1=train_00[train_00.columns.values[i]].plot(kind='kde',color='b',title=train_00.columns.values[i])
    f1=train_11[train_11.columns.values[i]].plot(kind='kde',color='y')
    f1=train_01[train_01.columns.values[i]].plot(kind='kde',color='r')
    f1=train_10[train_10.columns.values[i]].plot(kind='kde',color='orange')

    fig=f1.get_figure()
    fig.savefig('/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/'+train_00.columns.values[i]+'.pdf')






#t4=plt.plot(train_11[['V_Time']],kind='kde')#

help(pd.Series.plot)
type(train_00['V_Time'])






