# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:32:14 2021

@author: gao'x
"""
'''
四种模型——分类：1.分类、回归；2.线性、非线性；
学科的发展：线性→非线性
#
理论先导：深度学习
逻辑‘回归’：perceptron[感知机] 打包的逻辑回归过程
多个感知机线性加权输出like 电路板
输入层=>隐含层（神经元）=>输出层 ：多重感知机=>神经网络
每个神经元=一个g(ax+b)
神经网络、神经元各若干个 全连接的神经网络
参数：隐含层的个数 神经元性状 每个隐含层体现非线性

'''
#%% attention
#缺失值填充
import numpy as np
import pandas as pd
data=np.random.randn(5,3)
df=pd.DataFrame(data,columns=['one','two','three'])
df.iloc[1,2]=np.nan
df.iloc[0,1]=np.nan
#用某值填空值
df.fillna(0,,inplace=True)#or df1=df.fillna
#时间序列，沿用上一值填充
df2=df.fillna(method='ffill')
df3=df2.fillna(0)
#使用均值填空
mean=df['two'].mean()
df['two'].fillna(mean,inplace=True)

mean=df['three'].mean()
df['three'].fillna(mean,inplace=True)
#df.iloc[:,1].fillna(mean,inplace=True）

#%% neural network神经网络
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
#multi l process?

#设置谜底
sample_size=100
x=np.linspace(0,6,sample_size)
y=np.sin(x)
plt.plot(x,x1)

for i in range(10):
    size=(i + 1) * 100
    model=MLPRegressor(
        hidden_layer_sizes=(size,200))
    model.fit(x.reshape((sample_size,1)),y)
    Y=model.predict(x.reshape((sample_size,1)))
    
    plt.show()
    plt.scatter(x,Y)
    plt.scatter(x,y)

#猜谜
hidden_layer_sizes=(40,)
model=MLPRegressor(hidden_layer_sizes)
#relu 折线 y=0(x<0),Y=x(x>0) 可以构造所有非线性函数 很好 默认relu
model.fit(x.reshape((sample_size,1)),y)
Y=model.predict(x.reshape((sample_size,1)))
plt.scatter(x,Y)
plt.scatter(x,y)
plt.title(str(hidden_layer_sizes))
#%% 优化
#%% 画个圆
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
sample_size=10000
x=np.random.uniform(-10,10,(sample_size,2))
plt.scatter(x[:,0],x[:,1],color='orange')
radius=7
y=np.zeros(sample_size)
for i in range(sample_size):
    if np.sqrt(x[i,0]**2+x[i,1]**2) < radius :
        y[i]=1
plt.figure(figsize=(10,10))
plt.scatter(x[y==0,0],x[y==0,1],color='orange')
plt.scatter(x[y==1,0],x[y==1,1],color='pink')

#logisticregression
model=LogisticRegression()
model.fit(x,y)
model.score(x,y)

x_test=np.random.uniform(-10,10,(10000,2))
y_pred =model.predict(x_test)
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='orange')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x,y)
model.score(x,y)

x_test=np.random.uniform(-10,10,(10000,2))
y_pred =model.predict(x_test)
plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='orange')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')

#%% Dtree
import sklearn.tree as tree
model=tree.DecisionTreeClassifier()
model.fit(x,y)
model.score(x, y)

x_test=np.random.uniform(-10,10,(10000,2))
y_pred =model.predict(x_test)
plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='orange')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')

plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)  

#%% LRegress
#%% 全链接的神经网络 classification
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(1000,10))
model.fit(x,y)
model.score(x,y)

x_test=np.random.uniform(-10,10,(10000,2))
y_pred =model.predict(x_test)
plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='orange')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')


















