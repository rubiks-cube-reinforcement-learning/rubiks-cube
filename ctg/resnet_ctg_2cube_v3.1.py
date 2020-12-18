#!/usr/bin/env python
# coding: utf-8

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
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # setting GPU indices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

train_in = pd.read_csv('train_input.csv')
train_out = pd.read_csv('train_output.csv')

class NumpyDataset(Dataset):

    def __init__(self, input_file_name, output_file_name, is_train=True):
        X_in = pd.read_csv(input_file_name)
        X_in.drop(columns=['ID'], inplace=True)
        y_in = pd.read_csv(output_file_name)
        y_in.drop(columns=['ID'], inplace=True)
        X_in['distance'] = y_in['distance']
        self.len = len(X_in)

        X_train, X_test = train_test_split(X_in, test_size=.30, random_state=1)

        if is_train:
            X_train = X_train.reset_index()
            y = X_train['distance']
            X = X_train.drop(columns=['distance', 'index'])
        else:
            y = X_test['distance']
            X = X_test.drop(columns=['distance'])

        freq_dict = y.value_counts().to_dict()
        y_freq = y.apply(lambda x: 1./freq_dict[x])
        self.idx_weights = y_freq

        self.X = X.to_numpy()
        self.y = y.to_numpy()

    def __getitem__(self, index):

        X_i = self.X[index]
        y_i = self.y[index]

        return X_i, y_i

    def __len__(self):
        return len(self.y)

batch_size = 4096

train_dataset = NumpyDataset('train_input.csv', 'train_output.csv')
weighted_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.idx_weights, len(train_dataset), replacement=True)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler)
valid_dataset = NumpyDataset('train_input.csv', 'train_output.csv',
                             is_train=False)
testloader = DataLoader(valid_dataset, batch_size=batch_size)

class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int,
                 resnet_dim: int, num_resnet_blocks: int,
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
                self.blocks.append(
                    nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

        self.dummy = nn.Linear(self.state_dim * self.one_hot_depth, out_dim,
                               bias=True)

    def forward(self, states_nnet):
        # print('In Model: {}'.format(states_nnet.size()))  # checking if all GPUs are being used
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
        return x


net = ResnetModel(24, 6, 1000, 100, 4, 1, True)

net = nn.DataParallel(net, device_ids=[0,1,2,3])
net = net.to(device)

# loss
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)

def train(net, trainloader):
    i = 0
    for epoch in range(5):  # no. of epochs
        running_loss = 0
        for i, data in enumerate(trainloader):
            # data and labels to GPU if available

            inputs, labels = data[0].to(device, non_blocking=True) - 1, data[
                1].to(device, non_blocking=True)

            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs).float()

            loss = criterion(outputs.view(-1).float(), labels.view(-1).float())

            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()

            running_loss += loss.item()

            if not i % 500 and i > 0:
                print(running_loss / i)

            i += 1
        print('[Epoch %d] loss: %.3f' %
              (epoch + 1, running_loss / len(trainloader)))

    print('Done Training')


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True) - 1, data[
                1].to(device, non_blocking=True)
            outputs = net(inputs)
            # pdb.set_trace()
            #             _, predicted = torch.max(outputs.data, 1)
            #             print(outputs.data.shape)
            #             print(predicted.shape)
            total += labels.size(0)
            correct += (torch.round(
                outputs.data.reshape(-1)) == labels).sum().item()
    print('Accuracy of the network on test set: %0.3f %%' % (
                100 * correct / total))



train(net, trainloader)

#test(net, trainloader)

test(net, testloader)

