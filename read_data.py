import torch
import numpy as np 
import pandas as pd 
import torch.utils.data as Data

def z_score(x : np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()

    return (x - mean) / std

def get_chosen_data() -> dict:
    df = pd.read_csv('600519.csv')
    
    df.set_index(['日期'], inplace = True)
    df = df.iloc[::-1]

    x = df[['收盘价', '最高价', '最低价', '涨跌额', '成交量', '成交金额', '总市值']]
    target = df['涨跌幅']

    return {
        'x' : x,
        'target' : target
    }


def normize_data(data_dict : dict) -> dict:
    x = data_dict['x']
    columns = x.columns
    # print(x.head())

    for col_name in columns:
        x[col_name] = x[[col_name]].apply(z_score)
    
    data_dict['x'] = x.to_numpy()

    return data_dict


def get_pair_data(data_dict : dict, time_window : int) -> tuple:
    x_data = data_dict['x']
    y_data = data_dict['target']
    total_length = len(y_data)

    y_numpy = np.array(y_data[time_window :])
    y = torch.tensor(y_numpy.reshape((-1, 1))).type(torch.float32)
    
    x_numpy = np.array(
        [x_data[i : i + time_window].reshape(time_window, -1) for i in range(total_length - time_window)]
    )
    x = torch.tensor(x_numpy).type(torch.float32)
    
    return x, y

def get_pair_data_only(data_dict : dict, time_window : int) -> tuple:
    x_data = data_dict['x']
    y_data = data_dict['target']
    total_length = len(y_data)

    y_numpy = np.array(y_data[time_window :])
    y_only = np.zeros_like(y_numpy)
    where = y_numpy > 0
    y_only[where] = 1

    y = torch.tensor(y_only).type(torch.long)
    
    x_numpy = np.array(
        [x_data[i : i + time_window].reshape(time_window, -1) for i in range(total_length - time_window)]
    )
    x = torch.tensor(x_numpy).type(torch.float32)
    
    return x, y

def get_prepared_data(time_window : int) -> tuple:
    chosen_data_dict = get_chosen_data()
    normized = normize_data(chosen_data_dict)
    return get_pair_data(normized, time_window)

def get_prepared_data_only(time_window : int) -> tuple:
    chosen_data_dict = get_chosen_data()
    normized = normize_data(chosen_data_dict)
    return get_pair_data_only(normized, time_window)

def shuffle_data(x : torch.Tensor, y : torch.Tensor) -> tuple:
    index = np.arange(0, len(y))
    np.random.shuffle(index)

    return x[index], y[index]


if __name__ == '__main__':
    # test = get_prepared_data(3)

    # print(test[:5])

    # import matplotlib.pyplot as plt 
    # chosen = get_chosen_data()

    # plt.plot(chosen['target'].to_numpy())
    # plt.grid('on')

    # plt.title('每日涨跌幅度 (%)', fontproperties = 'SimHei')
    # # plt.tight_layout()
    # plt.savefig('涨跌幅度.png', dpi = 300)
    # plt.show()
    
    prepared_data = get_prepared_data(3)
    print(prepared_data[0][0].size())