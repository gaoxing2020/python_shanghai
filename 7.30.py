# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:06:43 2021

@author: gao'x
"""
#%% 数据类型 array&list
import numpy as np
'''
array
存储为数组
一维数组a   和  二维数组b 的区别
↑（3，）        ↑（3，1）
不可调用a(0,0)   可调用
'''
a=np.array([1,2,3])
b=np.array([[1,2,3]])

'''
list
存储为数据块，可伸缩
'''
c=[1,2,3]
e=[]    #创建空list
e.append(1) #添加元素
e.append(2)
e+e
e*4

#list←→array
d=np.array(e)
f=list(d)

#
g=np.ones(10)
h=g     #并非创建了两个数组，而是一个数组两个名字
h[0]=2  #h、g array同时改变
h2=g.copy()
h2[0]=6     #copy函数不改变原数组
#%% 读取csv数据 pandas
import pandas as pd

path=r"C:\Users\gao'x\Desktop\2020-2024\python\iris.csv"   
#需要将反斜杠转为正斜杠或加r,补充文件名
data=pd.read_csv(path,index_col=0)  #header=None,若无标题
data.columns
data_v=data.values

data.iloc[0,0]  #indexlocation
data.iloc[0,:]
data.iloc[:,-1]
#%% KNN model
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5,p=1)
#p=1为曼哈顿距离，p=2为欧几里得距离，n_neighbor为k值,默认值为5,knn好用但是运算量大，容易过拟合
x=data.iloc[:,0:4]
y=data.iloc[:,4]
model.fit(x,y)
model.score(x, y)

path_test=r"C:\Users\gao'x\Desktop\2020-2024\python\iris_test.csv" 
data_test=pd.read_csv(path_test,index_col=(0))
xtest=data_test.iloc[:,0:4]
x_test=np.array(xtest)
ytest=model.predict(x_test)

area1=data.iloc[:,0]*data.iloc[:,1]
area2=data.iloc[:,2]*data.iloc[:,3]
data['area1']=area1
data['area2']=area2     #在dataframe里添加数据列
data_final=data.iloc[[0,1,2,3,5,6,4],]

save_path=r"C:\Users\gao'x\Desktop\2020-2024\python\iris_new.csv"  
data_final.to_csv(save_path)
#%% 将样本本身分为样本内外
'''
样本内->train
样本外->test
python内置函数
'''
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.4,shuffle=(True))
#shuffle（洗牌）默认为true
model=KNeighborsClassifier(n_neighbors=1)
model.fit(train_x,train_y)
print(model.score(train_x,train_y))
print(model.score(test_x,test_y))
#%% python如何做优化
'''
运筹学硕士 optimazation
梯度下降 lamda learning rate
'''

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*x- x*2 - 5

def fderiv(x):
    return 2*x - 2
learning_rate=0.1
n_iter=100      #循环次数

xs=np.zeros(n_iter+1)
xs[0]=100

for i in range(n_iter):
    xs[i+1]=xs[i]-learning_rate*fderiv(xs[i])

plt.plot(xs)
#%% python已有的优化函数库
from scipy.optimize import minimize
#sci=scince
def f2(x):
    return  np.exp(-x*x)*(x**2)
f2(1)
x=np.linspace(-10,10,10000)
y=f2(x)
plt.plot(x,y)
minimize(f2,x0=-100).x

#%% entropy信息熵
import matplotlib.pyplo as plt
def E(x):
    a= -x*np.log(x) - (1-x)*np.log(1-x)
    return a
x=np.linspace(0.01,1,100)
y=E(x)
plt.plot (x,y)
#%% 决策树
'''
参数：
最高长几层？max_depth
如何避免过拟合？限制层数
没有分类就结束？少数服从多数（投票
仅针对离散标签嘛？
否，连续性也可：对数值范围进行分割
遍历特征找出使熵最小的分割点
信息学之父 香农 信息熵公式
信息熵最开始以2为底，（0、1）
'''
import pandas as pd

wm=r"C:\Users\gao'x\Desktop\wm_1.csv"
data=pd.read_csv(wm,index_col=0)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

import sklearn.tree as tree
model=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
#除了用信息熵即entropy衡量外，还可以用gini系数，entropy为以2为底的信息熵
model.fit(x,y)
model.score(x, y)
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)   #filled 选择填充
#%% 模型练习
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
bc=load_breast_cancer()
X=bc.data   #非dataframe，不用iloc
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.6)
model=tree.DecisionTreeClassifier('entropy',max_depth=(6))
model.fit(x_train,y_train)
model.score(x_train,y_train)
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)

model.fit(x_test,y_test)
model.score(x_test,y_test)  #1
plt.figure(figsize=(1,1))
tree.plot_tree(model,filled=True)

#%% knn训练此模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
bc=load_breast_cancer()
X=bc.data
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.6)
model=KNeighborsClassifier()
model.fit(x_train,y_train)
model.score(x_train,y_train) 

model.fit(x_test,y_test)
model.score(x_test,y_test) #0.93
#%% 逻辑 训练此模型
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
bc=load_breast_cancer()
X=bc.data
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.6)
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_train,y_train)

model.fit(x_test,y_test)
model.score(x_test,y_test)  #0.96

#%% 画个圆
import matplotlib.pyplot as plt
sample_size=100000
z=np.random.uniform(-10,10,(sample_size,2))
plt.scatter(z[:,0],z[:,1],color='orange')
radius=7
label=np.zeros(sample_size)
for i in range(sample_size):
    if np.sqrt(z[i,0]**2+z[i,1]**2) < radius :
        label[i]=1
plt.figure(figsize=(10,10))
plt.scatter(z[label==0,0],z[label==0,1],color='orange')
plt.scatter(z[label==1,0],z[label==1,1],color='pink')
