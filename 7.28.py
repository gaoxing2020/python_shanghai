# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:15:14 2021

@author: gao'x
"""
a = 1
b = 0.5
c = a + b
d = a*b
e = True
f = False

import numpy as np    #numberpython np可替代
import matplotlib.pyplot as plt  #绘图

arr=np.array([1,2,3]) #列向量，双击右侧variable explorer出现窗口
arr2=arr*2+1   #每个元素均作相同运算
arr/2

print(arr2) #print函数
print('你好python')

plt.plot(arr2) #图在右侧“plot”
plt.plot(arr)

arr[0]  #索引切片
arr[3]

arr_part = arr[:2]
print(arr_part)

#生成数组 zeros,ones,linspace(等差数列)函数
arr_0 = np.zeros(100)
print(arr_0)
arr_1 = np.ones(100)
print(arr_1)
a = np.linspace(0,1,1001)   #(start,end,差值)
plt.plot(a)

#数组之间的运算
a2 = np.linspace(1,2,1001)  #运算与矩阵运算相同（？
a+a2
a3= a*a2    #内积、点乘或对应相乘相加  ?
np.dot(a,a2)    #点乘
np.sum(a3)  #sum（）函数
a4 = a2 * a2-a2
print (a4)
plt.figure()    #防止多线进入一图   ?
plt.plot(a4)
a/a2    #注意除数是否包含零
a2/a

#%%                             #重新开始
#循环和判断
arr_0=np.zeros(100)
 #从0开始，循环100遍,indent->缩进,i 可以为任意值
for i in range(100): 
    arr_0[i]=i*i
    
#判断，首先初始化一个数组
x = np.zeros(100)
for i in range(100):
    if arr_0[i]<=50:
        x[i]=0
    else:
        x[i]=1

#%% 函数

y=1
np.sin(y)
np.exp(y)

#自定义函数
def f(y):
    z=y**2+np.sin(y)-np.exp(y)
    return(z)

f(1)

o=np.linspace(-1,1,101)
p=np.zeros(101)
for i in range(101):
    p[i]=f(o[i])

plt.plot(o,p)
plt.scatter(o,p)    #散点图

#简化版 why y处可以放向量？A:该函数针对向量成立
p2=f(o)
plt.plot(p2)

p==p2

1000%2     #模运算

np.log(10)    #自然对数








