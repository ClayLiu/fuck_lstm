import os
import numpy as np 

import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import KFold

from net import StockForecast_lstm
from read_data import get_prepared_data, shuffle_data

ground_path = 'epoch_k-fold\\find_num_layers\\'
training_path = ground_path + 'training\\'

# 创建实验结果保存的放置文件夹
if not os.path.exists(training_path):
    os.makedirs(training_path)

with open(training_path + 'done_count.txt', 'w') as f:
    f.write(str(0))

# 查看是否可以用 cuda
if_cuda = torch.cuda.is_available()
if if_cuda:
    device = torch.device('cuda:0')
    print('正在使用显卡炼丹。。')
else:
    print('正在烤机')

if_shut_down_str = input('是否跑完就关机？[Y/N]\n')
if_shutdown = 'Y' in if_shut_down_str or 'y' in if_shut_down_str

num_layers_array = np.arange(1, 10)
np.save(training_path + 'num_layers_array', num_layers_array)

# 超参数
k = 10              # 设置为 10 折验证
time_window = 5     # 时间窗为 5
batch_size = 16     # 批次大小 为 16
repeat_time = 3     # 重复 3 次实验
epoch_count = 100   # 训练 100 次

# 保存实验超参数
with open(training_path + '实验超参数.txt', 'w') as f:
    f.write('k = {}\n'.format(k))
    f.write('time_window = {}\n'.format(time_window))
    f.write('batch_size = {}\n'.format(batch_size))
    f.write('repeat_time = {}\n'.format(repeat_time))
    f.write('epoch = {}\n'.format(epoch_count))
    

x, y = get_prepared_data(time_window)  # 全部数据

for k_fold_time in range(repeat_time):
    shuffle_x, shuffle_y = shuffle_data(x, y)   # 将数据置乱

    this_num_layers_train_loss = []
    this_num_layers_test_loss = []

    kf = KFold(n_splits = k, shuffle = True)
    for num_layers in num_layers_array:
        print(num_layers)

        current_fold_time_train_loss = np.zeros(10)
        current_fold_time_test_loss = np.zeros(10)

        for i, (train_index, test_index) in enumerate(kf.split(shuffle_x)):
            
            # 准备训练数据 与 测试数据
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

            net = StockForecast_lstm(num_layers)
            
            if if_cuda:
                net.to(device)

            loss_function = nn.MSELoss()
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
            
            train_pred = net(train_x)
            this_time_train_loss = loss_function(train_pred, train_y).item()

            test_pred = net(test_x)
            this_time_test_loss = loss_function(test_pred, test_y).item()

            current_fold_time_train_loss[i] = this_time_train_loss
            current_fold_time_test_loss[i] = this_time_test_loss

        this_num_layers_train_loss.append(current_fold_time_train_loss.mean())
        this_num_layers_test_loss.append(current_fold_time_test_loss.mean())

        np.save(training_path + 'test_loss{}'.format(k_fold_time), np.array(this_time_test_loss))
        np.save(training_path + 'train_loss{}'.format(k_fold_time), np.array(this_time_train_loss))

        with open(training_path + 'done_count.txt', 'w') as f:
            f.write(str(k_fold_time + 1))

with open(training_path + 'finish_time.txt', 'w') as f:
    import datetime
    f.write(datetime.datetime.now().strftime('%H:%M:%S'))

# 将结果画图
# from get_only_result_images import *
# get_images(training_path)

if if_shutdown:
    os.system('shutdown -p')