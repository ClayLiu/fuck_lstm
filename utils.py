import torch
from torch import nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from net import get_pred, StockForecast_only_lstm

softmax = nn.Softmax()


def get_acc(network : StockForecast_only_lstm, x : torch.Tensor, y : torch.Tensor) -> float:
    pred = get_pred(network, x)
    return accuracy_score(y.detach().numpy(), pred.detach().numpy())

def get_precision(network : StockForecast_only_lstm, x : torch.Tensor, y : torch.Tensor) -> float:
    pred = get_pred(network, x)
    return precision_score(y.detach().numpy(), pred.detach().numpy())

def get_recall(network : StockForecast_only_lstm, x : torch.Tensor, y : torch.Tensor) -> float:
    pred = get_pred(network, x)
    return recall_score(y.detach().numpy(), pred.detach().numpy())

def get_f1_score(network : StockForecast_only_lstm, x : torch.Tensor, y : torch.Tensor) -> float:
    pred = get_pred(network, x)
    return f1_score(y.detach().numpy(), pred.detach().numpy())