import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets.samples_generator import make_blobs


import psutil
import os

import sys
import time


from sklearn.decomposition import PCA

#from sklearn.datasets.samples_generator import make_classification
#X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2,
 #                          n_clusters_per_class=1,class_sep =0.5, random_state =10)
start = time.clock()

filename = "data-p2p.txt"
scanin=np.loadtxt(filename)
bb=-48
ee=0
for i in range(32):
    bb=bb+48
    ee=ee+48
    X=scanin[bb:ee,[0,2,3,4]]
    y=scanin[bb:ee,7]


    pca = PCA(n_components=1)
    pca.fit(X)       
    X_new = pca.transform(X)
       
#    print(X,y)
   
#    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
#    ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)
    
#    lda = LinearDiscriminantAnalysis(n_components=1)
#    lda.fit(X,y)
#    X_new = lda.transform(X)
#    print(X_new)
    
#    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
#    ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],marker='o',c=y)
#    plt.scatter(X_new[:, 0], X_new[:, 1],X_new[:,2],marker='o',c=y)
    
#    plt.scatter(X[0:24,0],X[0:24,1],marker='o',color='r',s=10)
#    plt.scatter(X[25:48,0],X[25:48,1],marker='o',color='b',s=10)
#    
#    plt.scatter(100000*X_new[0:24, 0],100000*X_new[0:24, 0],marker='o',color='y')
#    plt.scatter(100000*X_new[25:48, 0],100000*X_new[25:48, 0],marker='o',color='g')
#    
    
    plt.show()
elapsed = (time.clock() - start)
print("Time used:",elapsed)

info = psutil.virtual_memory()
print ('内存占用',psutil.Process(os.getpid()).memory_info().rss)
print(sys.getsizeof(X_new))
#print ("总内存",info.total)
#print ('内存占比',info.percent)
#print ('cpu个数',psutil.cpu_count())