# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


import psutil
import os
#import sys

#数据类型
#7个维度+1个时间+1个ID，共65个ID
#目标：分割成小组 2个ID+2个维度的数据
#处理流程：先对每一个ID的数据，计算各个维度的相关性取均值，再分割数据

pd.set_option('display.max_columns',10)
#行可显示pd.set_option('display.max_colwidth',100)
start = time.clock()

#filename = "data-p2p.txt"
filename = "k-data-less.txt"

scanin=np.loadtxt(filename)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

h=[]
ii=0
jj=-1
kk=0
bb=-48
ee=0
for ii in range(8):
    ii=ii+1
    jj=jj+2
    kk=kk+2
    bb=bb+48
    ee=ee+48
 #   print type(ee)
    X=scanin[bb:ee,[0,2,7]]
    pos0=np.where(X[:,2]==jj) 
  #  print(pos0)
    pos1=np.where(X[:,2]==kk)
  #  print(pos1)
    X1=X[pos0,0:2]#pos代表位置，输出第三位为0下第0-2的值
    X1=X1[0,:,:]#位置全清0
    #print(X1)
 #   print("class0X1",X1,X1.shape,jj)#输出矩阵规格
    X2=X[pos1,0:2]
    X2=X2[0,:,:]#位置全清0
 #   print("class1X2",X2,X2.shape,kk)
    
    M1=np.mean(X1,0)#对X2各列求均值
    M1=np.array([M1]) #变为矩阵
    #print("M1X1rowmean",M1,M1.shape)
    M2=np.mean(X2,0)# ，0代表压缩行，对X2各列求均值，
    M2=np.array([M2])
    #print("M2X2rowmean",M2,M2.shape)
    M=np.mean(X[:,0:2],0)#对X各列求均值,print(X[:,0:2])
    M=np.array([M])
    #print(M)
    
    p=np.size(X1,0)#axis = 0，返回X1的行数 
    q=np.size(X2,0)#X2的行数
    #print(p,q),p为0类的数量，q为1类的数量
    
    #第二步，求类内散度矩阵
    S1=np.dot((X1-M1).transpose(),(X1-M1))#。dot为点积,减去均值进行归一化
    #print("S1:",S1)
    S2=np.dot((X2-M2).transpose(),(X2-M2))
    #print(S2)
    Sw=(p*S1+q*S2)/(p+q)
#    print(Sw)
    
        #第三步，求类间散度矩阵
    Sb1=np.dot((M1-M).transpose(),(M1-M))
#    print(Sb1,jj)
    Sb2=np.dot((M2-M).transpose(),(M2-M))
#    print(Sb2,kk)
    Sb=(p*Sb1+q*Sb2)/(p+q)
#    print(Sb)
    
    #判断Sw是否可逆
#    bb=np.linalg.det(Sw)
#    print(bb)
    
    
        #第四步，求最大特征值和特征向量
    [V,L]=np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
#    print("VVVVLLLL",V,L)#V是特征值，L是特征向量
    list1=[]
#    print("LIST11",list1 )
    a=V
    list1.extend(a)
#    print("list1111",list1)
    b=list1.index(max(list1))
#    print("bbbb",b)
    #print(a[b])
    W=L[:,b]#选出特征值大的特征向量
#    print("WWWWW", W)
    
    #根据求得的投影向量W画出投影线
    k=W[1]/W[0]#k=特征值1/特征值0
    h.append(k)
    b=0;
    x=np.arange(2,250)#2,3,4,5,6,7,8,9,10)
    yy=k*x+b
#    plt.plot(x,yy)
#    plt.scatter(X1[:,0],X1[:,1],marker='o',color='g',s=20)
#    plt.scatter(X2[:,0],X2[:,1],marker='o',color='b',s=20)
#    plt.grid()
#    
#    plt.show()
#    
    #计算第一类样本在直线上的投影点
    xi=[]
    yi=[]
    for i in range(0,p):
        y0=X1[i,1]#第二列
        x0=X1[i,0]#第一列
        x1=(k*(y0)+x0)/(k**2+1)
        
        y1=k*x1
        xi.append(x1)
        yi.append(y1)
#    print("first000x",np.mat(xi),jj)
#    print("first000y",yi,jj)


#相关性验证
#    X23=X[0:24,0]
#    X24=X[0:24,1]
#    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
#    print(X23)
#    print(X24)
#    print(xi)
    #  c = np.vstack((xi,X2))
#    X23=np.mat(X23)
#    xi=np.mat(xi)
#    X555=np.vstack((X23,xi))
#    X555=np.transpose(X555)
#    df = pd.DataFrame(X555)
#    Y=df.corr()
#    print(round(Y))

#    
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
#    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
#    print("sec000x",xj,kk)
#    print("sec000y",yj,kk)
    
    #相关性验证
#    X23=X[24:48,0]
#    X24=X[24:48,1]
# #   np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
# #   print(len(X23))
#    print(X24)
##    print(len(xj))
#    X23=np.mat(X23)
#    xj=np.mat(xj)
#    X555=np.vstack((X23,X24))
#    print(X555)
#    X555=np.transpose(X555)
#    df = pd.DataFrame(X555)
#    Y=df.corr()
#    print(Y)
    
    
    
    
    
#print("kkkkk",sorted(h))
#print("finish") 
    
    
    
    #画出投影后的点
    plt.plot(x,yy)
    cloo=np.array(['#FF7B73','#00008B','#00F007','#E0F80F', '#BD9D5C', '#00FFFF', '#7FFFD4', 
               '#6D6D6D', '#7D6105', '#FF9967', '#000000', '#62777A', '#0000FF', '#8A2BE2',
               '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED',
               '#6277BC', '#DC143C', '#00FFFF', '#008B8B', '#B8860B', '#A9A9A9', '#006400', 
               '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', 
               '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', 
               '#696969', '#1E90FF', '#B22222', '#825AA3', '#228B22', '#FF00FF', '#336699', 
               '#996699', '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#FF8D8A', 
               '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', 
               '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90'])
             
#    plt.scatter(X1[:,0],X1[:,1],marker='o',color=cloo[jj],s=3)#原始数据的点
#    plt.scatter(X2[:,0],X2[:,1],marker='o',color=cloo[kk],s=3)#原始数据的点
#    plt.grid()
#    plt.plot(xi,yi,color=cloo[jj],marker='+')
#    plt.plot(xj,yj,color=cloo[kk],marker='+')
#    
    plt.show()
    
    
    
print(h,len(h))
#h1=h
#h2=np.array(h1)
#print(np.argsort(h2))
#print(h2)
#h3=sorted(h2)
#h3=np.array(h3)
#print(h3)



elapsed = (time.clock() - start)
print("Time used:",elapsed)

#print(sys.getsizeof(xi))
info = psutil.virtual_memory()
print ('内存占用',psutil.Process(os.getpid()).memory_info().rss)
#print ("总内存",info.total)
#print ('内存占比',info.percent)
#print ('cpu个数',psutil.cpu_count())



