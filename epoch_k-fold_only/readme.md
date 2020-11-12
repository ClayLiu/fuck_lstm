这是使用 3次10折交叉验证得到的，根据训练次数 epoch 的不同所得到的 精度与 f1 指标。

图在 epoch_acc.png 与 epoch_f1.png 中。

数据在 epoch_acc_and_f1.npz 中。

该 npz 的定义如下表。

|    键值     |                        解释                        |
| :---------: | :------------------------------------------------: |
| epoch_array | 长度为95的数组，每个元素为某个指标所对应的训练轮数 |
|  train_f1   |                在训练集上得到的 f1                 |
|   test_f1   |                在测试集上得到的 f1                 |
|  train_acc  |             在训练集上得到的 预测精度              |
|  test_acc   |             在测试集上得到的 预测精度              |

所使用的的网络架构如下：
```python
class StockForecast_only_lstm(nn.Module):
    def __init__(self):
        super(StockForecast_only_lstm, self).__init__()

        self.lstm = nn.LSTM(
            input_size = 7,
            hidden_size = 32,
            num_layers = 2,
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
```
