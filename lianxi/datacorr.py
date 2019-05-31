import numpy as np
import pandas as pd
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

for i in range(64):
  X1=X[m:n,0:7]
  Y=pd.DataFrame(X1).corr()
#  df = pd.DataFrame(X1)
#  Y=df.corr()
  YY=YY+Y
 #print(X1,X1.shape)
  i=i+1
  m=m+24
  n=n+24

print(YY,YY.shape)aaa

#65组数据
#0和2,5和6，1和4
