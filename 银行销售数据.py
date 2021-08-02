# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:16:57 2021

@author: gao'x
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.tree as tree

#导入数据data
path=r"C:\Users\gao'x\Desktop\2020-2024\银行销售数据.csv"
data=pd.read_csv(path)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

model=tree.DecisionTreeClassifier(cretrion='entropy',max_depth=(4))
model.fit(x_train, y_train)
model.score(x_train,y_train)#1.0

model.score(x_test, y_test)#0.8875940762320952

tree.plot_tree(model,filled=True)

#Logistic
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
a=x_train[:,2]
model.fit(x_train[:,:2],y_train)
model.score(x_train, y_train)#0.9091602788664979

model.score(x_test, y_test)#0.9091602788664979
intercept_logic=model.intercept_ 
coef_logic=model.coef_

#Linear
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
model.score(x_train, y_train)#0.33723408034015545

model.score(x_test, y_test)#0.33903127827127966
intercept_linear=model.intercept_ 
coef_linear=model.coef_
coef_linear=coef_linear.reshape(1,-1)

#单个线性回归
from sklearn.linear_model import LinearRegression
model=LinearRegression()
a=np.array(x_train.iloc[:,14])
b=np.array(y_train)
model.fit(a.reshape((28831,1)),b.reshape((-1)))

interc=model.intercept_ 
coef=model.coef_

#logic
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
a=np.array(x_train.iloc[:,[0,1,3]])
b=np.array(y_train)
model.fit(a,b)
model.score(x_test.iloc[:,[0,1,3]],y_test)

interc=model.intercept_ 
coef=model.coef_
print(coef)
model.score(x_test.iloc[:,[0,1,2,3]],y_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5,p=1)
model.fit(x_train,y_train)
model.score(x_train, y_train)

model.score(x_test, y_test)


#决策树
import sklearn.tree as tree
from sklearn.tree import export_graphviz
x_train=x_train.iloc[:,[2,4,7]]
model=tree.DecisionTreeClassifier(criterion='entropy',max_depth=(3))
model.fit(x_train,y_train)
plt.figure(dpi=2000)
tree.plot_tree(model,filled=True) 








