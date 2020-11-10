import numpy as np 
import matplotlib.pyplot as plt 

import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import KFold

from net import StockForecast_lstm
from read_data import get_prepared_data, shuffle_data

k = 10

time_window = 5

batch_size = 16

x, y = get_prepared_data(time_window)       # 全部数据
shuffle_x, shuffle_y = shuffle_data(x, y)   # 置乱后数据

mean_test_loss_list = []
epoch_array = np.arange(1, 100 + 1)
np.save('epoch_array_6', epoch_array)
kf = KFold(n_splits = k, shuffle = True)
for epoch_count in epoch_array:
    this = np.zeros(3)
    for i in range(3):
        test_loss = []
        times = 0
        for train_index, test_index in kf.split(shuffle_x):
            times += 1
            print(times)

            train_x = shuffle_x[train_index]
            train_y = shuffle_y[train_index]

            test_x = shuffle_x[test_index]
            test_y = shuffle_y[test_index]

            train_dataset = Data.TensorDataset(train_x, train_y)
            train_data_loader = Data.DataLoader(
                dataset = train_dataset,
                batch_size = batch_size,
                shuffle = True
            )


            net = StockForecast_lstm()
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters(), weight_decay = 1e-5)

            for epoch in range(epoch_count):
                for step, (batch_x, batch_y) in enumerate(train_data_loader):
                    
                    optimizer.zero_grad()
                    pred = net(batch_x)

                    loss = loss_function(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                
            test_pred = net(test_x)
            loss = loss_function(test_pred, test_y)
            test_loss.append(loss.item())
        this[i] = np.array(test_loss).mean()

    mean_test_loss_list.append(this.mean())
    np.save('test_loss_for_epoch_6', np.array(mean_test_loss_list))



with open('time.txt', 'w') as f:
    import datetime
    f.write(datetime.datetime.now().strftime('%H:%M:%S'))

import os
os.system('shutdown -p')