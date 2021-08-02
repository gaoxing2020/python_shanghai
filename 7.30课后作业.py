# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:15:35 2021

@author: gao'x
"""
#%%1 ×
import numpy as np
def A(x,n):
    a=np.linspace(1,x,x)
    a=list(a)
    b=a[1:n+1]
    a=a.extend(b)
    for i in range(n):
        a.pop(0)
    return a#生成一行数量为x的数列,n为行数
A(5,3)
#%%1 解析
import numpy as np
n=5
mat= np.zeros((n,n))
mat[0,:]=np.arange(1,n+1,1)
for i in range(1,n):
    mat[i,n-1]=mat[i-1,0]
    for j in range(n-1):
        mat[i,j]=mat[i-1,j+1]
#法二
for i in range(n):
    for j in range(n):
        mat[i,j]=(i+j)%n+1

#%%2
def B(x):
    a=1
    for i in range(1,x+1):
        a = a * i
    return a
#%%3
#(1)
import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return x**4 - x**2
x=np.linspace(-1,1,1000) #缩小范围才能看出来中间的形状
y=f(x)
plt.plot(x,y)
#(2)
f(2)
#(3)
from scipy.optimize import minimize
res=minimize(f, x0=-2).x      #x0为初始值——
res1=minimize(f, x0=-0.5).x
res2=minimize(f, x0=0).x #0 斜率为0
res3=minimize(f, x0=1).x #0.707107
#%%4
from sklearn.datasets import load_boston
boston=load_boston()
x=boston.data
y=boston.target

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
model=LinearRegression()
x_trian,x_test,y_train,y_test=train_test_split(x,y,train_size=406)
'''
train_size：可以为浮点、整数或None，默认为None
①若为浮点时，表示训练集占总样本的百分比
②若为整数时，表示训练样本的样本数
③若为None时，train_size自动被设置成0.75
'''
model.fit(x_trian,y_train)
model.score(x_trian,y_train) #0.7452845944008999

model.score(x_test,y_test) #0.7045922043477976
#%%5
#(1)
import numpy as np
DS = np.random.binomial(1,0.5,(2000,10))#ds=dataset
Y=(DS[:,0]*DS[:,1]+DS[:,2]*DS[:,3]+DS[:,4]*DS[:,5]+DS[:,6]*DS[:,7]+DS[:,8]*DS[:,9])%2
#(2)
from sklearn.model_selection import train_test_split
ds_train=DS[-1000:,:]
ds_test=DS[:1001,:]
Y_train=Y[-1000:]
Y_test=Y[:1001]
#(3)
#逻辑回归
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(ds_train,Y_train)
model.score(ds_train,Y_train) #0.546

model.score(ds_test,Y_test) #0.48951048951048953
#KNN
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(ds_train,Y_train)
model.score(ds_train,Y_train) #0.821

model.score(ds_test,Y_test) #0.6763236763236763
#决策树
import sklearn.tree as tree
model=tree.DecisionTreeClassifier()
model.fit(ds_train,Y_train)
model.score(ds_train,Y_train) #1.0

model.score(ds_test, Y_test) #0.8881118881118881

'''
受问题本身的性质影响
'''













