## -*- coding: utf-8 -*-
#ip = 55
#ii=888
#for ii in range(11):
#    
#    print(ii,ip)
#    ip=ip+1
#  #  print(ii)
import matplotlib.pyplot as plt
import numpy as np


x=0
y=0
t=-1
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
print (len(cloo))
for i in range(60):
    x=x+1
    y=y+1
    t=t+1
    plt.plot(x,y)
    plt.scatter(x,y,marker='o',color=cloo[t],s=50)