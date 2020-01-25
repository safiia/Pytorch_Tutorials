import numpy as np
import torch
from time import perf_counter
from dlc_practical_prologue import *

(train_input,train_target,test_input,test_target)= load_data(one_hot_labels=True, normalize=True)

train_input.shape

train_target=train_target*0.9

eps=1e-6

def sigma_(x):
    return torch.tanh(x)

def dsigma(x):
     return (1- torch.pow(sigma_(x),2))

def loss(v,t):
    return torch.pow((t-v),2)

def dloss(v,t):
    return -2*(t-v)

w1=torch.empty(50, 784).normal_(0,eps)
w2=torch.empty(10, 50).normal_(0,eps)

b1 = torch.zeros(50)
b2 = torch.zeros(10)

dw1=torch.zeros(w1.size())
dw2=torch.zeros(w2.size())
db1=torch.zeros(b1.size())
db2=torch.zeros(b2.size())

def forward_pass(w1,b1,w2,b2,x):
    x0 = x
    s1 = torch.mv(w1,x0.view(-1))+b1
    x1 = sigma_(s1)
    s2 = torch.mv(w2,s1.view(-1,)) + b2
    x2 = sigma_(s2)
    return (x0, s1, x1, s2, x2)



def backward_pass(w1,b1,w2,b2,t,x,s1,x1,s2,x2,dw1,db1,dw2,db2):
    x0 = x
    dl_x2 = dloss(x2, t)
    dx2_s2 = dsigma(s2)
    
    dl_s2 = torch.mul(dl_x2,dx2_s2)
    dl_x1 = torch.mv(w2.t(),dl_s2)
    dx1_s1 = dsigma(s1)
    
    dl_s1 = torch.mul(dl_x1, dx1_s1)
    
#     db1=dl_s1
#     db2=dl_s2
    
    dl_dw1=torch.mm(dl_s1.view(-1,1),x0.view(1,-1))
    dl_dw2=torch.mm(dl_s2.view(-1,1),x1.view(1,-1))
    dw2.add_(dl_dw2)
    db2.add_(dl_s2)
    
    dw1.add_(dl_dw1)
    db1.add_(dl_s1)
    pass

#los = []
errors=0
   
dw1=torch.zeros(w1.size())
dw2=torch.zeros(w2.size())
db1=torch.zeros(b1.size())
db2=torch.zeros(b2.size())
for j in range(50):
    predict = torch.zeros(1000)
    tr = torch.zeros(1000)
    errors=0

   # lr = 0.1

    dw1.zero_()
    db1.zero_()
    dw2.zero_()
    db2.zero_()
    l =0
    for i in range(1000):
        
        x0, s1, x1, s2, x2 = forward_pass(w1,b1,w2,b2,train_input[i])
        predict = torch.argmax(x2)
        l = l+ loss(x2, train_target[i])
        
        #if train_target[i, predict] < 0.5: errors = errors + 1
       # tr[i] = torch.argmax(train_target[i])
        backward_pass(w1,b1,w2,b2,test_target[i],train_input[i],s1,x1,s2,x2,dw1,db1,dw2,db2)
        
    w1=w1-(0.0001*dw1)
    b1=b1-(0.0001*db1)
    w2=w2-(0.0001*dw2)
    b2=b2-(0.0001*db2)
#     templ = loss(torch.tensor(predict).view(-1,),torch.tensor(tr).view(-1,)).numpy()
       
#     los.append(templ)
    
    for k in range(1000):
        
        _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, test_input[k])
        #print(x2.size())
        predict = torch.argmax(x2)
        if test_target[k, predict] < 0.5:
        
            errors += 1

    #print('loss: ' + str(l))
    print ('accuracy =  ' + str(100*errors/1000))

    




