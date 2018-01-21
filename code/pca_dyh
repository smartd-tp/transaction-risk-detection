import pandas            as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from imblearn.combine import SMOTEENN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

train=pd.read_csv(r"C:\Users\Administrator\Desktop\大数据竞赛-风险识别算法赛\train.csv",delimiter=',')

pca = PCA(n_components=2)
x = np.array(train.iloc[:,2:32])
train_pca = pca.fit_transform(x)
train_pca_all = pd.DataFrame({'Id':list(train['ID']),
                              'Label':list(train['Label']),
                          'pca1':list(train_pca[:,0]),
                              'pca2':list(train_pca[:,1])})

# print(list.train_pca)

pd.DataFrame(train_pca).to_csv(r"C:\Users\Administrator\Desktop\大数据竞赛-风险识别算法赛\train_pca.csv")
# plt.scatter(train_pca[:,0],train_pca[:,1])

plt.scatter(train_pca_all[train_pca_all['Label']==1]['pca1'],train_pca_all[train_pca_all['Label']==1]['pca2'],color='r')
plt.scatter(train_pca_all[train_pca_all['Label']==0]['pca1'],train_pca_all[train_pca_all['Label']==0]['pca2'],color='b')



# with open(train_pca.txt,'w') as fl:
#     fl.write(train_pca)
#     fl.close()
