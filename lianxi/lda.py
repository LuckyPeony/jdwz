# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

X=np.loadtxt("22.txt")

pos0=np.where(X[:,2]==0) 
print(pos0)
pos1=np.where(X[:,2]==1)

print(pos1)

X1=X[pos0,0:2]
X1=X1[0,:,:]
print(X1,X1.shape)
X2=X[pos1,0:2]
X2=X2[0,:,:]

print(X2,X2.shape)

M1=np.mean(X1,0)
M1=np.array([M1])
print(M1,M1.shape)
M2=np.mean(X2,0)
M2=np.array([M2])
print(M2)
M=np.mean(X[:,0:2],0)
M=np.array([M])
print(M)

p=np.size(X1,0)
print(p)
q=np.size(X2,0)

print(q)

#第二步，求类内散度矩阵
S1=np.dot((X1-M1).transpose(),(X1-M1))
print(S1)
S2=np.dot((X2-M2).transpose(),(X2-M2))
print(S2)
Sw=(p*S1+q*S2)/(p+q)

print(Sw)

#第三步，求类间散度矩阵
Sb1=np.dot((M1-M).transpose(),(M1-M))
print(Sb1)
Sb2=np.dot((M2-M).transpose(),(M2-M))
print(Sb2)
Sb=(p*Sb1+q*Sb2)/(p+q)

print(Sb)

#判断Sw是否可逆

bb=np.linalg.det(Sw)

print(bb)

#第四步，求最大特征值和特征向量
[V,L]=np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
print(V,L.shape)
list1=[]
a=V
list1.extend(a)
print(list1)
b=list1.index(max(list1))
print(a[b])
W=L[:,b]

print(W,W.shape)

#根据求得的投影向量W画出投影线
k=W[1]/W[0]
b=0;
x=np.arange(2,10)#2,3,4,5,6,7,8,9,10
yy=k*x+b
plt.plot(x,yy)
#plt.scatter(X1[:,0],X1[:,1],marker='o',color='g',s=20)
#plt.scatter(X2[:,0],X2[:,1],marker='o',color='b',s=20)
plt.grid()

plt.show()

#计算第一类样本在直线上的投影点
xi=[]
yi=[]
for i in range(0,p):
    y0=X1[i,1]
    x0=X1[i,0]
    x1=(k*(y0-b)+x0)/(k**2+1)
    y1=k*x1+b
    xi.append(x1)
    yi.append(y1)
print(xi)

print(yi)

#计算第二类样本在直线上的投影点
xj=[]
yj=[]
for i in range(0,q):
    y0=X2[i,1]
    x0=X2[i,0]
    x1=(k*(y0-b)+x0)/(k**2+1)
    y1=k*x1+b
    xj.append(x1)
    yj.append(y1)
print(xj)

print(yj)

#画出投影后的点
plt.plot(x,yy)
#plt.scatter(X1[:,0],X1[:,1],marker='o',color='g',s=20)
#plt.scatter(X2[:,0],X2[:,1],marker='o',color='b',s=20)
plt.grid()
plt.plot(xi,yi,'g+')
plt.plot(xj,yj,'b+')

plt.show()
