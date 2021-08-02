# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 08:53:12 2021

@author: gao'x
"""
#%%线性回归
#线性回归
#创造样本
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #***

sample_size=11
x_arr=np.linspace(0,10,sample_size)    
'''
 linspace是Matlab中的均分计算指令，用于产生x1,x2之间的N点行线性的矢量。其中x1、x2、
 N分别为起始值、终止值、元素个数。若默认N，默认点数为100。
'''
slope=2
intercept=1
y_arr=x_arr*slope+intercept

#创造噪音
std=1   #标准差
epsilon=np.random.normal(0,std,sample_size)#random函数库正态分布

y_arr=y_arr+epsilon     #数组元素对应相加，要求元素数量相同
plt.scatter(x_arr,y_arr)

#最小二乘法
model=LinearRegression()
model.fit(x_arr.reshape(sample_size,1),y_arr.reshape(-1))     #拟合,数列必为列向量，因此需要reshape（矩阵行，矩阵列）
a=model.coef_
b=model.intercept_

z_arr=np.linspace(0,1,100)
z_predict=model.predict(z_arr.reshape(100,1))
plt.figure()
plt.scatter(z_arr, z_predict)

#二维（高维）线性回归   a1\a2\a3\a4\b???
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
model_iris=LinearRegression()
model_iris.fit(x,y)
z=np.array([[5.6,3,6,2],[4.6,3.1,2,0.5]])
z_predict=model_iris.predict(z)

#classification分类学习
def sigmoid(x):
    y=1/(np.exp(-x)+1)
    return y

x=np.linspace(-10,10,101)
y=sigmoid(x)
plt.plot(x, y)

#%%逻辑回归
#制作样本
sample_size=100000

x=np.random.uniform(0,10,(sample_size,2))#均匀分布;生成100个二维坐标
x1=x[:,0]
x2=x[:,1]   #第一处（100个数）全都取，第二处只取第一个数
plt.scatter(x1, x2)
y=np.zeros(sample_size)
z=np.random.normal(0,0,sample_size)
for i in range(sample_size):
    if x2[i]>x1[i]+z[i]:
        y[i]=0
    else:
        y[i]=1

#取出y=0的点
y1_index=(y==1)
y0_index=(y==0)
plt.scatter(x1[y1_index],x2[y1_index],color='green')
plt.scatter(x1[y0_index],x2[y0_index],color='orange')

#logical regression逻辑回归（其实不是回归是分类->线性分类
#调取模型 blackbox？
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x,y)
print(model.intercept_) #fit之后才能调取
print(model.coef_)
x_test=np.array([
    [10,0],
    [0,10],
    [5,2],
    ])
y_test_pred=model.predict(x_test)
#样本外
xt1=np.arange(-0.1,10.1,0.1) #arange函数
xt2=np.arange(-0.1,10.1,0.1)
xxt1,xxt2=np.meshgrid(xt1,xt2) #meshgrid混合函数
plt.scatter(xxt1.reshape(-1),xxt2.reshape(-1),color='grey')#-1意味着将方阵展平

xxt1.shape #查看形态

xt=np.hstack([xxt1.reshape(-1,1),xxt2.reshape(10404,1)])#hstack：横向并列
yt=model.predict(xt)
plt.scatter(xt[yt==0,0],xt[yt==0,1],color='orange')
plt.scatter(xt[yt==1,0],xt[yt==1,1],color='green')

#任何一个模型fit之后都会由一定错误
#如何衡量？
model.score(x, y)

#%%iris小练习
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
model=LogisticRegression()
model.fit(x,y)
z=np.array([[5.6,3,6,2],
            [4.6,3.1,2,0.5]])
z_predict=model.predict(z)
model.score(x, y)

#%%reshape
x=np.zeros(100)
x_1=x.reshape(25,4)     #按照原数组顺序，按行排列
x_2=x.reshape(-1)   #一维向量
