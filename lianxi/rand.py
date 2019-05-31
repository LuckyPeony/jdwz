import random
#生成随机数，浮点类型
import os
import psutil
#import time

#start = time.clock()
for i in range(30000):
    a = random.randint(10, 80)
    b = random.randint(20,40)
#控制随机数的精度round(数值，精度)
#    print(a,b)


for i in range(30):
    c = random.randint(50, 90)
    d = random.randint(30,50)
#控制随机数的精度round(数值，精度)
#    print(c,d)
info = psutil.virtual_memory()
print ('内存占用',psutil.Process(os.getpid()).memory_info().rss)
#elapsed = (time.clock() - start)
#print("Time used:",elapsed)