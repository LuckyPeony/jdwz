import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys

#数据类型
#7个维度+1个时间+1个ID，共65个ID
#目标：分割成小组 2个ID+2个维度的数据
#处理流程：先对每一个ID的数据，计算各个维度的相关性取均值，再分割数据

pd.set_option('display.max_columns',10)
#行可显示pd.set_option('display.max_colwidth',100)


filename = "data-p2p.txt"
X=np.loadtxt(filename)
#print(X)
YY=np.zeros((7,7))
#print(YY,YY.shape)
i=0
m=0
n=24
T=0
for i in range(64):
  X1=X[m:n,0:7]
  df = pd.DataFrame(X1)
  Y=df.corr()
  YY=YY+Y
 #print(X1,X1.shape)
  i=i+1
  m=m+24
  n=n+24
  T=T+1

print(YY,YY.shape)
print(T)
#df = pd.DataFrame(X)
#Y=df.corr()
#print(Y)

#pos0=np.where(X[:,2]==0) 
##print(pos0)
#pos1=np.where(X[:,2]==1)
##print(pos1)
#X1=X[pos0,0:2]#pos代表位置，输出第三位为0下第0-2的值
#X1=X1[0,:,:]#位置全清0
##print(X1)
#print("class0X1",X1,X1.shape)#输出矩阵规格
#X2=X[pos1,0:2]
#X2=X2[0,:,:]#位置全清0
#print("class1X2",X2,X2.shape)

