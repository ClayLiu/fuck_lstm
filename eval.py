import torch
import matplotlib.pyplot as plt 

from net import StockForecast_lstm
from read_data import get_prepared_data, get_chosen_data

time_window = 3
epoch_count = 1000
trainning_rate = 0.7

test = StockForecast_lstm()
test.load_state_dict(torch.load('trainning\\trained_nets\\stock_try_{}_{}_{}.pkl'.format(time_window, epoch_count, int(trainning_rate * 100))))
test.eval()

x, y = get_prepared_data(time_window)


pred = test(x).detach().numpy()

plt.plot(y)
plt.plot(pred)
plt.xlabel('day')
plt.grid('on')

plt.legend(
    [
        'real change rate',
        'predicted change rate'
    ]
)


plt.show()