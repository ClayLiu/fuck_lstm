import numpy as np 
import matplotlib.pyplot as plt 

import torch
from torch import nn
import torch.utils.data as Data

from net import StockForecast_lstm
from read_data import get_prepared_data

time_window = 3
epoch_count = 10
batch_size = 16
trainning_rate = 0.7    # 使用全部数据的 70% 作为训练数据

x, y = get_prepared_data(time_window)   # 全部数据

whole_data_count = len(y)
train_data_count = int(whole_data_count * trainning_rate)  

train_x, train_y = x[: train_data_count], y[: train_data_count] # 训练数据
test_x, test_y = x[train_data_count : ], y[train_data_count : ] # 测试数据


train_dataset = Data.TensorDataset(train_x, train_y)
train_data_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)


net = StockForecast_lstm()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), weight_decay = 1e-5)

train_loss_history_list = []
test_loss_history_list = []
for epoch in range(epoch_count):
    for step, (batch_x, batch_y) in enumerate(train_data_loader):
        
        optimizer.zero_grad()
        pred = net(batch_x)

        loss = loss_function(pred, batch_y)
        loss.backward()
        optimizer.step()

    # if epoch % 10 == 0:
    print('epoch : {}, train loss : {}'.format(epoch, loss.item()))    
    train_loss_history_list.append(loss.item())

    test_pred = net(test_x)
    loss = loss_function(test_pred, test_y)
    print('epoch : {}, test loss : {}'.format(epoch, loss.item()))
    test_loss_history_list.append(loss.item())


train_loss_history = np.array(train_loss_history_list)
test_loss_history = np.array(test_loss_history_list)

loss_path = 'trainning\\{}_{}_{}_loss\\'.format(time_window, epoch_count, int(trainning_rate * 100))
import os
if not os.path.exists(loss_path):
    os.mkdir(loss_path)

np.save(loss_path + 'train_loss', train_loss_history)
np.save(loss_path + 'test_loss', test_loss_history)

net.eval()
torch.save(net.state_dict(), 'trainning\\trained_nets\\stock_try_{}_{}_{}.pkl'.format(time_window, epoch_count, int(trainning_rate * 100)))