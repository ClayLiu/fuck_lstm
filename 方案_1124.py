import os
import numpy as np 

import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import KFold

from net import StockForecast_only_lstm_num_layers
from read_data import get_prepared_data_only, shuffle_data
from utils import get_acc, get_f1_score

ground_path = 'epoch_k-fold_only\\num_layers=9\\'
training_path = ground_path + 'training_2\\'

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

epoch_array = np.arange(100, 5000, 100)
np.save(training_path + 'epoch_array', epoch_array)

# 超参数
k = 10              # 设置为 10 折验证
time_window = 5     # 时间窗为 5
batch_size = 16     # 批次大小 为 16
repeat_time = 3     # 重复 3 次实验

# 保存实验超参数
with open(training_path + '实验超参数.txt', 'w') as f:
    f.writelines('k = {}\n'.format(k))
    f.writelines('time_window = {}\n'.format(time_window))
    f.writelines('batch_size = {}\n'.format(batch_size))
    f.writelines('repeat_time = {}\n'.format(repeat_time))
    

x, y = get_prepared_data_only(time_window)  # 全部数据

for k_fold_time in range(repeat_time):
    shuffle_x, shuffle_y = shuffle_data(x, y)   # 将数据置乱

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

    kf = KFold(n_splits = k, shuffle = True)
    for epoch_count in epoch_array:
        print(epoch_count)
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

            # 使用有 9 层 lstm 隐藏层的网络
            net = StockForecast_only_lstm_num_layers(9)
            
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
        np.save(training_path + 'test_acc_for_epoch_{}'.format(k_fold_time), np.array(mean_test_acc_list))
        np.save(training_path + 'train_acc_for_epoch_{}'.format(k_fold_time), np.array(mean_train_acc_list))

        mean_test_f1_list.append(this_test_f1.mean())
        mean_train_f1_list.append(this_train_f1.mean())
        np.save(training_path + 'test_f1_for_epoch_{}'.format(k_fold_time), np.array(mean_test_f1_list))
        np.save(training_path + 'train_f1_for_epoch_{}'.format(k_fold_time), np.array(mean_train_f1_list))

        with open(training_path + 'done_count.txt', 'w') as f:
            f.write(str(k_fold_time + 1))

with open(training_path + 'finish_time.txt', 'w') as f:
    import datetime
    f.write(datetime.datetime.now().strftime('%H:%M:%S'))

# 将结果画图
from get_only_result_images import *
get_images(training_path)

if if_shutdown:
    os.system('shutdown -p')