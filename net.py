import torch
from torch import nn

class StockForecast_lstm(nn.Module):
    def __init__(self, num_layers = 2):
        super(StockForecast_lstm, self).__init__()

        self.lstm = nn.LSTM(
            input_size = 7,
            hidden_size = 32,
            num_layers = int(num_layers),
            batch_first = True
        )
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, hidden_cell = self.lstm(x, None)
        droped = self.drop_out(lstm_out[:, -1, :])
        pred = self.linear(droped)
        return pred

class StockForecast_only_lstm(nn.Module):
    def __init__(self, num_layers = 2):
        super(StockForecast_only_lstm, self).__init__()

        self.lstm = nn.LSTM(
            input_size = 7,
            hidden_size = 32,
            num_layers = int(num_layers),
            batch_first = True
        )
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(32, 2)
        # self.soft_max = nn.Softmax()

    def forward(self, x):
        lstm_out, hidden_cell = self.lstm(x, None)
        droped = self.drop_out(lstm_out[:, -1, :])
        pred = self.linear(droped)
        # pred = self.soft_max(linear)
        
        return pred


def get_pred(network : StockForecast_only_lstm, x : torch.Tensor) -> torch.Tensor:
    '''
        计算预测标签
    '''
    origin_pred = network(x)
    return torch.argmax(origin_pred, dim = 1)

if __name__ == '__main__':
    test = StockForecast_lstm()
    print(test)
