#!/usr/bin/env python
# coding: utf-8

# In[293]:


import graphviz
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[294]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ### Step 1. 在此填写要处理的文件名（.csv），以及保存的输出文件名
# 
# filename - 数据文件名  
# export_tree_name - 导出的决策树文件名（.pdf）  
# export_result_name - 导出的训练结果（.txt）  

# In[295]:


filename = 'randomall-for all data.csv'
export_tree_name = 'qxdtree4'
export_result_name = 'qxdtree4'


# In[296]:


# import data and try utf-8 and gbk decode
try:
    with open(filename, 'r', encoding='utf-8') as f:
        features = f.readline().split(',')[0:-1]
        data = np.loadtxt(f, skiprows=0, delimiter=',', dtype=np.float64)
        print('use utf-8 decoder\n------')
except UnicodeDecodeError:
    with open(filename, 'r', encoding='gbk') as f:
        features = f.readline().split(',')[0:-1]
        data = np.loadtxt(f, skiprows=0, delimiter=',', dtype=np.float64)
        print('use gbk decoder\n------')
except BaseException as e:
    print(e)

X = data[:,:-1]
Y = data[:,-1]
print('number of samples : {}\t\t\tnumber of features: {}\n------'.format(*X.shape))
print('Features and labels of the first 5 samples:\n')
print(' '.join(features))
for i in range(5):
    print(X[i, :], Y[i])
print('\n------')


# ### Step 2. 划分数据集
# 
# test_size - 数字，范围为0.0-1.0，代表测试集中包含数据集的比例  
# shuffle - 布尔值，True/False，决定划分数据前是否要打乱数据排序  

# In[297]:


from imblearn.over_sampling import SMOTE

X, Y = SMOTE(). fit_sample(X, Y)
from collections import Counter

print(sorted(Counter(Y).items()),
    sorted(Counter(Y).items()))


# In[298]:


'''
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0) #采用随机过采样（上采样）

X,Y=ros.fit_sample(X, Y)
'''


# In[299]:


'''
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(ratio=1,random_state=0,replacement=True) #采用随机欠采样（下采样）
X,Y = rus.fit_sample(X,Y)
'''


# In[300]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, stratify=Y)

print('sample number of train set: {}\nsample number of test set : {}'.format(y_train.shape[0], y_test.shape[0]))


# ### Step 3. 训练随机森林

# In[301]:


# define randomforest
clf = RandomForestClassifier(n_estimators=100, 
                             max_depth=100,
                             criterion='gini', 
                             max_features='sqrt', 
                             bootstrap=True, 
                             n_jobs=-1, 
                             oob_score=True)

# train randomforest 
clf.fit(X_train, y_train)


# ### Step 4. 输出训练结果

# In[302]:


# output OOB accurary
print('Random forest train result\n------')
print('out of bag (OOB) accurary: {}\n------'.format(clf.oob_score_))

# output feautre importance info
importance = {}
feature_importances = clf.feature_importances_
for i in range(X.shape[1]):
    importance[features[i]] = feature_importances[i]
print('feature importance:\n')
for t in sorted(importance.items(), key=lambda x:x[1], reverse=True):
    print('{:<20s} {:<.2f}'.format(*t))
print('\n------')

# validate model with test set and plot non-normalized confusion matrix
y_pred = clf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names=[ 'A1','A2','A3','A4','A5','A6',]

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()


# In[303]:


# # export decision tree into pdf file
# def tree_export(model):
#     dot_data = tree.export_graphviz(model, out_file=None)
#     graph = graphviz.Source(dot_data)
#     graph.render("dt")

# tree_export = (export_tree_name)


# In[304]:


# # define decision tree
# dt = tree.DecisionTreeClassifier(max_features="sqrt", max_depth=5)
# dt.fit(X, Y)


# In[305]:


# importance = {}
# feature_importances = dt.feature_importances_
# for i in range(X.shape[1]):
#     importance[features[i]] = feature_importances[i]

# # print importance order
# print('Decision Tree\nfeature importance:\n')
# for t in sorted(importance.items(), key=lambda x:x[1], reverse=True):
#     print('{:<20s} {:>.2f}'.format(*t))
    
# Y_pred = dt.predict(X)
# cnf_matrix = confusion_matrix(Y, Y_pred)
# np.set_printoptions(precision=2)
# class_names=['bad', 'good']

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()


# In[ ]:




