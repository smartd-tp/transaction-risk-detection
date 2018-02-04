# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pandas as pd


# load data 
#train_ori=np.loadtxt(open("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv","rb"),
#                     delimiter=",",skiprows=1)

#test_ori = np.loadtxt(open("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv","rb"),
#                     delimiter=",",skiprows=1)
train_ori=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/train.csv")
test_ori=pd.read_csv("/Users/cp/Documents/job/smartdec/AL_COMPETITION/workfile/pred.csv")

#### set up data , no labels, outlier and normal data are separated
train_norm=train_ori[train_ori["Label"]==0]
train_norm2 = np.array(train_norm.iloc[:,1:32])

train_outlier=train_ori[train_ori["Label"]==1]
train_outlier2=np.array(train_outlier.iloc[:,1:32])


modelSVM = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1) 
modelSVM.fit(train_norm2)     #30mins
SVM_pred_train = modelSVM.predict(train_norm2)   #10mins
SVM_pred_outlier = modelSVM.predict(train_outlier2)

n_error_train = SVM_pred_train[SVM_pred_train == -1].size
n_error_outliers = SVM_pred_outlier[SVM_pred_outlier == 1].size








xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))








#### fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()






















###################   LOF
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

# Generate train data
X = 0.3 * np.random.randn(100, 2)
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]


#####  setup data : contain normanl and outlier, but no labels
train_mix = np.array(train_ori.iloc[:,2:32])
temp=train_ori["Label"]==1
outlier_idx=np.array(temp[temp==1].index).tolist() ##outlier index

np.array(train_ori[train_ori["Label"]==1])

# fit the model
modelLOF=LocalOutlierFactor(n_neighbors=20,p=1,contamination=0.003)
lof_pred_train=modelLOF.fit_predict(train_mix)   
pred_outlier=np.array(np.where(lof_pred_train==-1)[0]).tolist()   ##predicted outlier

tp=list(set(outlier_idx).intersection(set(pred_outlier)))  # true positive




# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
y_pred_outliers = y_pred[200:]

# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Local Outlier Factor (LOF)")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

a = plt.scatter(X[:200, 0], X[:200, 1], c='white',
                edgecolor='k', s=20)
b = plt.scatter(X[200:, 0], X[200:, 1], c='red',
                edgecolor='k', s=20)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()















#############################isolation forest

from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)


train_norm=train_ori[train_ori["Label"]==0]
train_norm2 = np.array(train_norm.iloc[:,2:32])

train_outlier=train_ori[train_ori["Label"]==1]
train_outlier2=np.array(train_outlier.iloc[:,2:32])




# fit the model
modelIF = IsolationForest(max_samples=10000, max_features=5,random_state=rng,contamination=0.003)
modelIF.fit(train_norm2)
IF_pred_train = modelIF.predict(train_norm2)
IF_pred_outliers = modelIF.predict(train_outlier2)


n_error_train = IF_pred_train[IF_pred_train == -1].size
n_error_outliers = IF_pred_outliers[IF_pred_outliers == 1].size






################## random　 forest
from sklearn.ensemble import RandomForestClassifier

train_feature= np.array(train_ori.iloc[:,1:31])
train_tag=np.array(train_ori.iloc[:,32])
modelRF = RandomForestClassifier(max_depth=5, random_state=0)
modelRF.fit(train_feature, train_tag)

RF_pred_train=modelRF.predict(train_feature)

pred_outlier=np.array(np.where(RF_pred_train==1)[0]).tolist()   ##predicted outlier

tp=list(set(outlier_idx).intersection(set(pred_outlier)))  # true positive






