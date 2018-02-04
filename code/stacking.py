
# coding: utf-8

# # stacking算法#

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import scipy
from vecstack import stacking
import vecstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn import metrics

from sklearn.grid_search import GridSearchCV


# In[2]:


# 更改默认路径
import os
os.chdir(r"C:\Users\wudandan\Desktop\阿里云项目\大数据竞赛-风险识别算法赛")   #修改当前工作目录
os.getcwd()


# In[3]:


#读取数据
train=pd.read_csv("train.csv",delimiter=',')

train_xy_0,test_xy_0 = train_test_split(train,test_size = 0.3,random_state=1)
y_train_0 = train_xy_0.Label
X_train_0 = train_xy_0.drop(['Label'],axis=1)
y_test_0 = test_xy_0.Label
X_test_0 = test_xy_0.drop(['Label'],axis=1)


# In[23]:


pd.value_counts(y_train)


# In[29]:


#smote生成数据集
sm = SMOTE(ratio={1:24000},k_neighbors=2,kind='borderline2')
x_train=train_xy_0.iloc[:,2:-1]
y_train=train_xy_0.loc[:,'Label']
X_resampled, y_resampled = sm.fit_sample(x_train,y_train)
train_smo = pd.DataFrame(X_resampled,columns=list(x_train.columns))
train_smo['Label']=y_resampled

#train_smo.to_csv("train_smo.csv")


# In[30]:


#stacking建模#
#xgboost
params={
'booster':'gbtree',#运用提升树模型#
'objective': 'binary:logistic',#多分类的问题#
#'num_class':2, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。#
'max_depth':5, # 构建树的深度，越大越容易过拟合
'reg_lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。#
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
#'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'learning_rate': 0.1, # 如同学习率
'seed':0,
#'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
'n_jobs':-1,
'n_estimators':100

}

                 
#train_xy,test_xy = train_test_split(train_smo,test_size = 0.3,random_state=1)
y_train = train_smo.Label
X_train = train_smo.drop(['Label'],axis=1)
y_test = y_test_0
X_test = X_test_0.drop(['ID','V_Time'],axis=1)
#将数据做成矩阵形式
xgb_test = xgb.DMatrix(X_test,label=y_test)
xgb_train = xgb.DMatrix(X_train, label=y_train)
watchlist = [(xgb_train, 'train'),(xgb_test, 'test')]  


# # 调整XGBClassifier参数
# 

# In[8]:


#生成自定义函数
target = 'Label'
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
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
 
        #Print model report:
        print("\nModel Report")
        print("recall: %.4g" % metrics.recall_score(dtrain[target].values, dtrain_predictions))
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
        f1=2*recall_score(dtrain[target],dtrain_predictions)*accuracy_score(dtrain[target], dtrain_predictions)/(recall_score(dtrain[target],dtrain_predictions)+accuracy_score(dtrain[target],dtrain_predictions))
        print('f1_score: [%.8f]' % f1)
        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        #feat_imp.plot(kind='bar', title='Feature Importances')
        #plt.ylabel('Feature Importance Score')


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=150,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=0)

modelfit(xgb1,train_smo, predictors)


# In[31]:


predictors = [x for x in train_smo.columns if x !='Label' ]
target = 'Label'


# In[60]:


#step1：调整max_depth，min_child_weight

param_test1 = {
 'max_depth':list(range(3,10,2)),
 'min_child_weight':list(range(1,6,2))
}
for  score in ['f1_macro','f1']:
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=150, max_depth=5,
    min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4,scale_pos_weight=1, seed=0), 
     param_grid = param_test1,scoring=score,n_jobs=-1,iid=False, cv=5)
    gsearch1.fit(X_train,y_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[62]:


#step2：再次调整max_depth，min_child_weight
param_test1 = {
 'max_depth':list(range(4,7)),
 'min_child_weight':list(range(4,7))
}
for  score in ['f1_macro','f1']:
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=150, max_depth=6,
    min_child_weight=6, gamma=0, subsample=0.8,colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4,scale_pos_weight=1, seed=0), 
     param_grid = param_test1,scoring=score,n_jobs=-1,iid=False, cv=5)
    gsearch1.fit(X_train,y_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[63]:


gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[64]:


#step3:再次调整最小子节点个数
param_test1 = {
 'min_child_weight':list(range(7,15))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=150, max_depth=6,
     min_child_weight=6, gamma=0, subsample=0.8,colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4,scale_pos_weight=1, seed=0), 
     param_grid = param_test1,scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[71]:


#step4：gamma参数调优
param_test4 = {
 'gamma':[i/10.0 for i in range(0,6)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=150, max_depth=6,
 min_child_weight=12, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0), 
 param_grid = param_test4, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch4.fit(X_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[72]:


#step5:subsample,colsample_bytree参数调整
param_test5 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=150, max_depth=6,
 min_child_eight=12, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0), 
 param_grid = param_test5, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch5.fit(X_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[74]:


#step5:subsample,colsample_bytree参数调整
param_test5 = {
 'subsample':[i/100.0 for i in range(65,80,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=150, max_depth=6,
 min_child_weight=12, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0), 
 param_grid = param_test5, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch5.fit(X_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[14]:


#step6:正则化参数调优
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=150, max_depth=6,
 min_child_weight=12, gamma=0, subsample=0.7, colsample_bytree=0.85,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0), 
 param_grid = param_test6, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[ ]:


param_test7 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=150, max_depth=6,reg_alpha=1e-5,
 min_child_weight=12, gamma=0, subsample=0.7, colsample_bytree=0.85,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0), 
 param_grid = param_test7, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch7.fit(train[predictors],train[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


# In[ ]:


#调整种树的数量
param_test8 = {
 'n_estimators':list(range(100,2000,200))
}
gsearch8= GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=150, max_depth=6,
 min_child_weight=12, gamma=0, subsample=0.7, colsample_bytree=0.85,reg_alpha=1e-05,reg_lambda=1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0), 
 param_grid = param_test8, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch8.fit(train[predictors],train[target])
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_






# In[15]:


#降低学习速率
param_test9 = {
 'learning_rate':[0.001,0.1,0.05]
}
gsearch9= GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=500, max_depth=6,
 min_child_weight=12, gamma=0, subsample=0.7, colsample_bytree=0.85,reg_alpha=1e-05,reg_lambda=1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0,), 
 param_grid = param_test9, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch9.fit(train[predictors],train[target])
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_


# # 二、逻辑回归参数调整

# In[31]:


param_test9 = {
 'penalty':['l1','l2'],
'tol':[1e-5,1e-4,1e-3,0.01,0.05,0.1],
'C':[0.01,0.1,0.5,1,5,10,100]
}
gsearch9= GridSearchCV(estimator = LogisticRegression(penalty='l2', tol=1e-5,C=0.1,random_state=0,max_iter=100), 
 param_grid = param_test9, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch9.fit(train[predictors],train[target])
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_


# In[33]:


param_test9 = {'max_iter':[100,500,1000,3000]}
gsearch9= GridSearchCV(estimator = RandomForestClassifier(random_state = 0, n_jobs = -1, 
n_estimators = 300, max_depth = 6)
gsearch9.fit(train[predictors],train[target])
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_


# # 第三步、随机森林参数调整

# In[7]:


#1.调整深度及最小子节点样本
param_rfm1 = {
 'max_depth':list(range(3,10,2)),
 'min_samples_leaf':list(range(1,10,2))

}
gsearch_rfm1= GridSearchCV(estimator =  RandomForestClassifier(random_state = 0, n_jobs = -1, 
            n_estimators = 150, max_depth = 6),
             param_grid = param_rfm1, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch_rfm1.fit(train[predictors],train[target])
gsearch_rfm1.grid_scores_, gsearch_rfm1.best_params_, gsearch_rfm1.best_score_


# In[9]:


#2.调整分裂准则
param_rfm2 = {
'criterion':['gini','entropy']

}
gsearch_rfm2= GridSearchCV(estimator =  RandomForestClassifier(random_state = 0, n_jobs = -1, min_samples_leaf=1,
            n_estimators = 150, max_depth = 5),
             param_grid = param_rfm2, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch_rfm2.fit(train[predictors],train[target])
gsearch_rfm2.grid_scores_, gsearch_rfm2.best_params_, gsearch_rfm2.best_score_

    
    
    
    


# In[11]:


#3.调整max_featuresmin_impurity_decrease
param_rfm3 = {
 'max_features':[0.8],
 'min_impurity_decrease':[0.00001,0.0001,0.0005,0.001]
}
gsearch_rfm3= GridSearchCV(estimator =  RandomForestClassifier(random_state = 0, n_jobs = -1,min_samples_leaf= 1,criterion='entropy' , 
            n_estimators = 150, max_depth = 5),
             param_grid = param_rfm3, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch_rfm3.fit(train[predictors],train[target])
gsearch_rfm3.grid_scores_, gsearch_rfm3.best_params_, gsearch_rfm3.best_score_


# In[ ]:


#4.调整树的数量
param_rfm4 = {
 'n_estimators':[100,150,200,300,500,1000]
}
gsearch_rfm4= GridSearchCV(estimator =  RandomForestClassifier(random_state = 0, n_jobs = -1,min_samples_leaf=1 ,criterion='entropy',
                                                               max_features= 0.8,
            min_impurity_decrease=1e-5 ,n_estimators = 150, max_depth = 5),
             param_grid = param_rfm4, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch_rfm4.fit(train[predictors],train[target])
gsearch_rfm4.grid_scores_, gsearch_rfm4.best_params_, gsearch_rfm4.best_score_




# # stacking

# In[32]:


# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.
models = [
       LogisticRegression(penalty='l2', tol=1e-5,C=0.1,random_state=0,max_iter=100),
           
        
        RandomForestClassifier(random_state = 0, n_jobs = -1,min_samples_leaf=1 ,criterion='entropy',
                            max_features= 0.8,min_impurity_decrease= 1e-5,n_estimators =300 , max_depth = 5),
        
        XGBClassifier(learning_rate =0.1, n_estimators=500, max_depth=6,
        min_child_weight=6, gamma=0, subsample=0.7, colsample_bytree=0.85,reg_alpha=1e-05,reg_lambda=1e-05,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=0,booster='gbtree')]


# In[11]:


# Compute stacking features
S_train, S_test = stacking(models, np.array(X_train), np.array(y_train), np.array(X_test),regression = False,
                           metric = recall_score, n_folds = 4, 
                           stratified = True, shuffle = True, random_state = 0, verbose = 2)
# Initialize 2-nd level model
model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 3,booster='gbtree')
# Fit 2-nd level model
model = model.fit(S_train, y_train)
# Predict
y_pred = model.predict(S_test)
# Final prediction score
print('Final recall_score: [%.8f]' % recall_score(y_test, y_pred))    
print('Final accuracy_score: [%.8f]' % accuracy_score(y_test, y_pred))    
# Initialize 2-nd level model
model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 100, max_depth = 3)
# Fit 2-nd level model
model = model.fit(S_train, y_train)



#预测测试数据集
y_pred = model.predict(S_test)
# Final prediction score
f1=2*recall_score(y_test, y_pred)*accuracy_score(y_test, y_pred)/(recall_score(y_test, y_pred)+accuracy_score(y_test, y_pred))
print('recall_scoree: [%.8f]' % recall_score(y_test, y_pred))    
print('accuracy_score: [%.8f]' % accuracy_score(y_test, y_pred))    
print('f1_score: [%.8f]' % f1)
#混淆矩阵
a_y_test=np.array(y_test)
y_t_p=np.c_[a_y_test,y_pred]
frame_Y=pd.DataFrame(y_t_p,columns=['y_test','y_pred'])
frame_Y.groupby([frame_Y['y_test'],frame_Y['y_pred']]).size()


# In[24]:





# In[25]:


#预测给定的测试集
#读取数据
pred=pd.read_csv("pred.csv",delimiter=',')
X_test2=pred.drop(['ID','V_Time'],axis=1)


# Compute stacking features
S_train, S_test = stacking(models, np.array(X_train), np.array(y_train), np.array(X_test2),regression = False,
                           metric = recall_score, n_folds = 10, 
                           stratified = True, shuffle = True, random_state = 0, verbose = 2)


# Initialize 2-nd level model
model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 3,booster='gbtree')
# Fit 2-nd level model
model = model.fit(S_train, y_train)
# Predict
y_pred = model.predict(S_test)


# In[ ]:


#预测给定的测试集
#读取数据
pred=pd.read_csv("pred.csv",delimiter=',')
X_test2=pred.drop(['ID','V_Time'],axis=1)


# Compute stacking features
S_train, S_test = stacking(models, np.array(X_train), np.array(y_train), np.array(X_test2),regression = False,
                           metric = recall_score, n_folds = 10, 
                           stratified = True, shuffle = True, random_state = 0, verbose = 2)


# Initialize 2-nd level model
model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 3,booster='gbtree')
# Fit 2-nd level model
model = model.fit(S_train, y_train)
# Predict
y_pred = model.predict(S_test)


# In[17]:


type(y_pred)


# In[28]:


pd.value_counts(pd.DataFrame(y_pred,columns=['Label']).Label)


# In[20]:


y_pred_frame=pd.DataFrame(y_pred,columns=['Label'])
y_pred_frame['ID']=pred['ID']
y_pred_frame.to_csv('pred_frame.csv')


# In[54]:


#预测所有数据集
# Predict
y_test_all = train.Label
X_test_all = train.drop(['Label','ID','V_Time'],axis=1)

# Compute stacking features
scoring=['precision_macro','recall_macro']
S_train, S_test = stacking(models, np.array(X_train), np.array(y_train),
                           np.array(X_test_all,
                                    regression = False, 
                                    metric = recall_score, 
                                    n_folds = 10,
                                    stratified = True, 
                                    shuffle = True,
                                    random_state = 0, 
                                    verbose = 2)


# In[52]:


#预测给定的测试集
#读取数据
pred=pd.read_csv("pred.csv",delimiter=',')
X_test2=pred.drop(['ID','V_Time'],axis=1)


# Compute stacking features
S_train, S_test = stacking(models, np.array(X_train), np.array(y_train), np.array(X_test2),regression = False,
                           metric = recall_score, n_folds = 10, 
                           stratified = True, shuffle = True, random_state = 0, verbose = 2)


# Initialize 2-nd level model
model = XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 3)
# Fit 2-nd level model
model = model.fit(S_train, y_train)
# Predict
y_pred = model.predict(S_test)


# In[53]:


pd.value_counts(pd.DataFrame(y_pred,columns=['Label']).Label)


# In[ ]:


#调整smote参数
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection as ms
from sklearn import datasets, metrics, tree

from imblearn import over_sampling as os
from imblearn import pipeline as pl

print(__doc__)

RANDOM_STATE = 42

scorer = metrics.make_scorer(metrics.cohen_kappa_score)

# Generate the dataset
X, y = datasets.make_classification(n_classes=2, class_sep=2,
                                    weights=[0.1, 0.9], n_informative=10,
                                    n_redundant=1, flip_y=0, n_features=20,
                                    n_clusters_per_class=4, n_samples=5000,
                                    random_state=RANDOM_STATE)
smote = os.SMOTE(random_state=RANDOM_STATE)
cart = tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
pipeline = pl.make_pipeline(smote, cart)

param_range = range(1, 11)
train_scores, test_scores = ms.validation_curve(
    pipeline, X, y, param_name="smote__k_neighbors", param_range=param_range,
    cv=3, scoring=scorer, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(param_range, test_scores_mean, label='SMOTE')
ax.fill_between(param_range, test_scores_mean + test_scores_std,
                test_scores_mean - test_scores_std, alpha=0.2)
idx_max = np.argmax(test_scores_mean)
plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
            label=r'Cohen Kappa: ${0:.2f}\pm{1:.2f}$'.format(
                test_scores_mean[idx_max], test_scores_std[idx_max]))

plt.title("Validation Curve with SMOTE-CART")
plt.xlabel("k_neighbors")
plt.ylabel("Cohen's kappa")

# make nice plotting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
plt.xlim([1, 10])
plt.ylim([0.4, 0.8])

plt.legend(loc="best")
plt.show()


# In[36]:


#test
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
#导入数据，将数据分成测试集和验证集
data=pd.read_csv('train.csv')
train_xy,val = train_test_split(data, test_size = 0.3,random_state=1)
y = train_xy.Label
X = train_xy.drop(['Label'],axis=1)
val_y = val.Label
val_X = val.drop(['Label'],axis=1)

