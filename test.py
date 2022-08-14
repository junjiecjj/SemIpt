# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>



import numpy as np

import math


#普通阶乘
def fact(n):
    if n == 0:
        return 1
    else:
        return n*fact(n-1)
#普通Cmn
def Cmn(n,m):
    return fact(n)/(fact(n-m)*fact(m))



N = 100
p = 0.25

A = []
B = []
C = []


Hx = p*math.log()























