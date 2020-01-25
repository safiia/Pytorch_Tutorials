import torch
from torch.autograd import Variable #torch.autograd provides classes and functions implementing automatic 
#differentiation of arbitrary scalar valued functions
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    
train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

train_input, train_target = Variable(train_input), Variable(train_target)
    
model, criterion = Net(), nn.MSELoss()


#Q1:
def train_model(model, train_input, train_target, mini_batch_size):
    

    for e in range(0, 25):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            #print(output)
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)
    

train_model(model, train_input, train_target, 100)

#Q2: 

    
def compute_nb_errors(model, input_, target, mini_batch_size):
    sum_loss = 0
    errors = 0
    # We do this with mini-batches
    for b in range(0, input_.size(0), mini_batch_size):
        output = model(input_.narrow(0, b, mini_batch_size))
       # print(output.size())
        #print(output[b])
        #predict = torch.argmax(output,dim=1)
        for i in range(mini_batch_size):
           # print(target[i, predict])
            predict = torch.argmax(output[i])
            if target[i, predict] < 0.5:        
                 errors += 1
            
    print (errors)

compute_nb_errors(model, test_input, test_target, 100)

model = Net()
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target, 100)

#Q3:

class Net2(nn.Module):
    def __init__(self, hidden_unites):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, hidden_unites)
        self.fc2 = nn.Linear(hidden_unites, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
model = Net2(hidden_unites=10)
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target,100)
    
model = Net2(hidden_unites=50)
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target,100)
    

model = Net2(hidden_unites=200)
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target,100)
    

model = Net2(hidden_unites=500)
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target,100)
    
    
model = Net2(hidden_unites=1000)
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target,100)
    

    
#Q4: modify the net2 class by adding another convolution layer

class Net_new(nn.Module):
    def __init__(self,hidden_unites):
        super(Net_new, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, hidden_unites)
        self.fc2 = nn.Linear(hidden_unites, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        return x

model = Net_new(hidden_unites=1000)
for i in range(10):
    train_model(model, train_input, train_target, 100)
    compute_nb_errors(model, test_input, test_target,100)
    
    
    
    
    