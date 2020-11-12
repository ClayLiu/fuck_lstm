import numpy as np 
import matplotlib.pyplot as plt 

import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import KFold


from net import StockForecast_only_lstm
from read_data import get_prepared_data_only, shuffle_data
from utils import get_acc, get_f1_score


if_cuda = torch.cuda.is_available()
if if_cuda:
    device = torch.device('cuda:0')
    print('正在使用显卡炼丹。。')
else:
    print('正在烤机')

k = 10
time_window = 5
batch_size = 16
for k_fold_time in range(1, 3):
    x, y = get_prepared_data_only(time_window)  # 全部数据
    shuffle_x, shuffle_y = shuffle_data(x, y)   # 置乱后数据

    # 准确率 列表
    mean_test_acc_list = []
    mean_train_acc_list = []

    # f1 列表
    mean_test_f1_list = []
    mean_train_f1_list = []

    # 准确率 数组
    this_test_acc = np.zeros(10)
    this_train_acc = np.zeros(10)

    # f1 数组
    this_test_f1 = np.zeros(10)
    this_train_f1 = np.zeros(10)


    epoch_array = np.arange(1, 100 + 1)
    np.save('epoch_array_9', epoch_array)


    kf = KFold(n_splits = k, shuffle = True)
    for epoch_count in epoch_array:
        print(epoch_count)
        for i, (train_index, test_index) in enumerate(kf.split(shuffle_x)):

            train_x = shuffle_x[train_index]
            train_y = shuffle_y[train_index]

            test_x = shuffle_x[test_index]
            test_y = shuffle_y[test_index]

            if if_cuda:
                test_x, test_y = test_x.to(device), test_y.to(device)

            train_dataset = Data.TensorDataset(train_x, train_y)
            train_data_loader = Data.DataLoader(
                dataset = train_dataset,
                batch_size = batch_size,
                shuffle = True
            )


            net = StockForecast_only_lstm()
            
            if if_cuda:
                net.to(device)

            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), weight_decay = 1e-5)

            # 训练过程
            for epoch in range(epoch_count):
                for step, (batch_x, batch_y) in enumerate(train_data_loader):
                    
                    if if_cuda:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    pred = net(batch_x)

                    loss = loss_function(pred, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 计算准确率
            this_test_acc[i] = get_acc(net, test_x, test_y)
            this_train_acc[i] = get_acc(net, train_x, train_y)

            # 计算f1
            this_test_f1[i] = get_f1_score(net, test_x, test_y)
            this_train_f1[i] = get_f1_score(net, train_x, train_y)


        mean_test_acc_list.append(this_test_acc.mean())
        mean_train_acc_list.append(this_train_acc.mean())
        np.save('test_acc_for_epoch_{}'.format(k_fold_time), np.array(mean_test_acc_list))
        np.save('train_acc_for_epoch_{}'.format(k_fold_time), np.array(mean_train_acc_list))

        mean_test_f1_list.append(this_test_f1.mean())
        mean_train_f1_list.append(this_train_f1.mean())
        np.save('test_f1_for_epoch_{}'.format(k_fold_time), np.array(mean_test_f1_list))
        np.save('train_f1_for_epoch_{}'.format(k_fold_time), np.array(mean_train_f1_list))


with open('time.txt', 'w') as f:
    import datetime
    f.write(datetime.datetime.now().strftime('%H:%M:%S'))

import os
os.system('shutdown -p')