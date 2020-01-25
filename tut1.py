import numpy as np
import torch
from time import perf_counter
from dlc_practical_prologue import *
#Q1:Multiple views of a storage
matrix=torch.full((13,13), 1)
matrix[[1,6,11],:]=2

matrix[:,[1,6,11]]=2

matrix[3:5,3:5]=3

matrix[3:5,-5:-3]=3
matrix[8:10,8:10]=3
matrix[8:10,3:5]=3

#Q2:Flops per second
mat1=torch.empty(5000, 5000).normal_(0,3)
mat2=torch.empty(5000, 5000).normal_(0,3)

t1_start = perf_counter() 
res=torch.mm(mat1,mat2)
t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds:", 
                                        t1_stop-t1_start) 

#Q3 Playing with strides:

def mul_row(m):

    for i in torch.arange(len(m)):
        m[i] = m[i] * (i+1)
    return m
# number of operation/execution time




t1_start = perf_counter()
m=torch.full((1000,400),2)
res=mul_row(m)
t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds:", 
                                        t1_stop-t1_start)
res



def mul_row_fast(m):
   
    for i in torch.arange(len(m)):
        m[i] = torch.mul(m[i], (i+1))

    return m


t1_start = perf_counter()
m=torch.full((1000,400),2)
res=mul_row(m)
t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds:", 
                                        t1_stop-t1_start)
res

