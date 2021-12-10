# -*- coding: utf-8 -*-
"""
# **Simple Network**
"""

# train using GPU, if not available on your machine, use google colab.
import pandas as pd;
from scipy.stats import zscore
import torch as torch;

import numpy as np

import torchvision.datasets as datasets
from torchvision import transforms

import torch.nn as nn;
import torch.nn.functional as F;
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#read in the dataset
num_classes=10;

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
                             transform=transform )
full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                             transform=transform )



batch_size=64;

trainloader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(full_test_dataset, batch_size=batch_size,shuffle=False)

# create a neural network (inherit from nn.Module)
class ConvNetWithBatchNorm(nn.Module):
    # architecture of the network is specified in the constructor
    def __init__(self): 
        super(ConvNetWithBatchNorm, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),         
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(num_features=12)           
        )
        self.features1 = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)   
        )
        self.classifier = nn.Sequential(
            nn.Linear(12*6*6, 50),         
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(50,num_classes)            
        )
        
    # here we specify the computation (forward phase of training) how "x" is transfered into output "y"
    def forward(self, x):
        x = self.features(x)
        x = self.features1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x)

    # constructor and forward() - that is all we need, the rest is implemented in the nn.Module and we inherit it

# create an instance of the network
model=ConvNetWithBatchNorm().to(device);
criterion = F.nll_loss;


# this optimizer will do gradient descent for us
# experiment with learning rate and optimizer type
learning_rate = 0.001;

# note that we have to add all weights&biases, for both layers, to the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# we add a learning rate scheduler, which will modify the learning rate during training
# will initially start low, then increase it ("warm up"), and then gradually descrease it
n_epochs = 30;
num_updates = n_epochs*int(np.ceil(len(trainloader.dataset)/batch_size))
print(num_updates)


warmup_steps=1000;
def warmup_linear(x):
    if x < warmup_steps:
        lr=x/warmup_steps
    else:
        lr=max( (num_updates - x ) / (num_updates - warmup_steps), 0.)
    return lr;
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear);


val_loss = []
val_acc = []
train_loss = []
train_acc = []


for i in range(n_epochs):

    for j, data in enumerate(trainloader):
      
        inputs, labels = data        
        inputs=inputs.to(device);
        labels=labels.to(device);
        
        optimizer.zero_grad();

        #forward phase - predictions by the model
        outputs = model(inputs);
        #forward phase - risk/loss for the predictions
        risk = criterion(outputs, labels);
  
        # calculate gradients
        risk.backward();
        
        # take the gradient step
        optimizer.step();
        scheduler.step();


        batch_risk=risk.item();


    correct, loss_t = 0, 0;
    with (torch.no_grad()):
      for j, data in enumerate(testloader):
        
          inputs, labels = data        
          inputs=inputs.to(device);
          labels=labels.to(device);
          outputs = model(inputs);
          loss_t += criterion(outputs, labels)

          pred = outputs.data.max(dim=1, keepdim=True)[1]
          
          correct += pred.eq(labels.data.view_as(pred)).sum().item();
    val_acc.append(correct / len(testloader.dataset))
    val_loss.append(loss_t)


    correct, loss_t = 0, 0;
    with (torch.no_grad()):
      for j, data in enumerate(trainloader):
        
          inputs, labels = data        
          inputs=inputs.to(device);
          labels=labels.to(device);
          outputs = model(inputs);
          loss_t += criterion(outputs, labels)

          pred = outputs.data.max(dim=1, keepdim=True)[1]
          correct += pred.eq(labels.data.view_as(pred)).sum().item();

    train_loss.append(loss_t)
    train_acc.append(correct / len(trainloader.dataset))

    print("Epoch : {}  val_loss = {}  train_loss = {}  val_acc = {}  train_acc = {}".format(i+1, val_loss[-1], train_loss[-1], 
                                                                                            val_acc[-1], train_acc[-1]))

# training loss vs validation loss

plt.figure(figsize=(12,5), dpi=100)
plt.plot(range(1, n_epochs+1), val_loss, label='validation loss')
plt.plot(range(1, n_epochs+1), train_loss, label='train loss')
plt.title("training loss vs validation loss")
plt.legend(loc='upper left', fontsize=8)
plt.show()

# training accuracy vs validation accuracy

plt.figure(figsize=(12,5), dpi=100)
plt.plot(range(1, n_epochs+1), train_acc, label='train_accuracy')
plt.plot(range(1, n_epochs+1), val_acc, label='test_accuracy')
plt.title('training accuracy vs validation accuracy')
plt.legend(loc='upper left', fontsize=8)
plt.show()





"""# **Training From Scratch**"""

model = models.resnet18(pretrained=False).to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

print(model)

valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainloader) 

for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')

    for batch_idx, (data_, target_) in enumerate(trainloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0


    with torch.no_grad():
        model.eval()
        for data_t, target_t in (testloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(testloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
    model.train()



plt.figure(figsize=(12,5), dpi=100)
plt.plot(range(1, n_epochs+1), val_loss, label='validation loss')
plt.plot(range(1, n_epochs+1), train_loss, label='train loss')

plt.legend(loc='upper left', fontsize=8)
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(range(1, n_epochs+1), train_acc, label='train_accuracy')
plt.plot(range(1, n_epochs+1), val_acc, label='test_accuracy')
plt.title('predicted vs actual')




"""# **Transfer Learning**"""
model = models.resnet18(pretrained=True).to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

print(model)

valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainloader) 

for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')

    for batch_idx, (data_, target_) in enumerate(trainloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0


    with torch.no_grad():
        model.eval()
        for data_t, target_t in (testloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(testloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
    model.train()
    

plt.figure(figsize=(12,5), dpi=100)
plt.plot(range(1, n_epochs+1), val_loss, label='validation loss')
plt.plot(range(1, n_epochs+1), train_loss, label='train loss')

plt.legend(loc='upper left', fontsize=8)
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(range(1, n_epochs+1), train_acc, label='train_accuracy')
plt.plot(range(1, n_epochs+1), val_acc, label='test_accuracy')
plt.title('predicted vs actual')


