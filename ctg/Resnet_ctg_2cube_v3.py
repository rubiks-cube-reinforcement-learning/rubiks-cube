#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


print(device)


# In[4]:


train_in = pd.read_csv('train_input.csv')
train_out = pd.read_csv('train_output.csv')


# In[5]:


train_in.head()


# In[6]:


train_out.head()


# In[35]:


class NumpyDataset(Dataset):

    def __init__(self, input_file_name, output_file_name, is_train=True):
        X_in = pd.read_csv(input_file_name)[:10000]
        X_in.drop(columns=['ID'], inplace=True)
        y_in = pd.read_csv(output_file_name)[:10000]
        y_in.drop(columns=['ID'], inplace=True)
        X_in['distance'] = y_in['distance']
        self.len = len(X_in)

        X_train, X_test = train_test_split(X_in, test_size=.01, random_state=1)

        if is_train:
            X = X_train
        else:
            X = X_test

        self.class_holder = dict()
        for i in range(1, 15):
            subsetX = X[X['distance'] == i].copy()
            subsetX.drop(columns=['distance'], inplace=True)
            self.class_holder[i] = subsetX.to_numpy()

    def __getitem__(self, index):

        choice_idx = np.random.choice(range(1, 15))

        x = self.class_holder[choice_idx]
        while len(x) == 0:
            choice_idx = np.random.choice(range(1, 15))
            x = self.class_holder[choice_idx]

        x_choice = x[np.random.choice(range(len(x)))]

        return x_choice, choice_idx

    def __len__(self):
        return self.len


# In[ ]:





# In[ ]:





# In[36]:


train_dataset = NumpyDataset('train_input.csv', 'train_output.csv')
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
valid_dataset = NumpyDataset('train_input.csv', 'train_output.csv', is_train=False)
testloader = DataLoader(valid_dataset, batch_size=64)


# In[22]:


class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

        self.dummy = nn.Linear(self.state_dim * self.one_hot_depth, out_dim, bias=True)
        
    def forward(self, states_nnet):
        x = states_nnet
        
        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()
        
        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        
        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        
        x = F.relu(x)
        
        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)
        
            x = F.relu(x + res_inp)
        
        # output
        x = self.fc_out(x)
#         x = F.one_hot(x.long(), self.one_hot_depth)
#         x = x.view(-1, self.state_dim * self.one_hot_depth)
#         x = x.float()
#         x= self.dummy(x)+11
        return x


# In[23]:


net = ResnetModel(24, 6, 1000, 100, 4, 1, True)
net = net.to(device)


# In[24]:


# loss
criterion = nn.MSELoss()
#critetion= nn.L1Loss()
# optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)


# In[25]:


# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         outputs = self.linear(x.float())
#         return outputs.float()


# In[26]:


# net = LogisticRegression(24, 1)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
# net = net.to(device)


# In[27]:


# for x, y in trainloader:
#     print(x-1, x.shape)
#     print(y, y.shape)
#     break
# outputs = net(x-1)
# print(outputs, outputs.shape)


# In[28]:


def train(net, trainloader):
    i = 0
    for epoch in range(10):  # no. of epochs
        running_loss = 0
        for i, data in enumerate(trainloader):
            # data and labels to GPU if available

            inputs, labels = data[0].to(device, non_blocking=True) - 1, data[1].to(device, non_blocking=True)
#             print(inputs.shape, labels.shape)
            # pdb.set_trace()

            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs).float()
#             print(outputs.shape)
            loss = criterion(outputs.view(-1).float(), labels.view(-1).float())
            
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()

            running_loss += loss.item()

            if not i % 500 and i > 0:
                print(loss.item())
#                 print(running_loss / i)

            i += 1
        print('[Epoch %d] loss: %.3f' %
              (epoch + 1, running_loss / len(trainloader)))
#               (epoch + 1, running_loss / len(trainloader)))

    print('Done Training')


# In[29]:


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True) - 1, data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            # pdb.set_trace()
#             _, predicted = torch.max(outputs.data, 1)
#             print(outputs.data.shape)
#             print(predicted.shape)
            total += labels.size(0)
            correct += (torch.round(outputs.data.reshape(-1)) == labels).sum().item()
    print('Accuracy of the network on test set: %0.3f %%' % (100 * correct / total))


# In[39]:


train(net, trainloader)


# In[40]:


test(net, trainloader)


# In[41]:


test(net, testloader)


# In[ ]:





# In[ ]:




