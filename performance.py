import NN
import RNN
import LSTM
import GRU
import torch
import torch.nn as nn


def model_data(model, data_type):
    if model == "NN":
        model, path, train_loader, test_loader = NN.summary_return()
    elif model == "RNN":
        model, path, train_loader, test_loader = RNN.summary_return()
    elif model == "LSTM":
        model, path, train_loader, test_loader = LSTM.summary_return()
    elif model == "GRU":
        model, path, train_loader, test_loader = GRU.summary_return()
    else:
        print("load one of the models name please")
    if data_type == "train":
        loader = train_loader
    elif data_type == "test":
        loader = test_loader
    else:
        print("select train or test please")

    def load_checkpoint(checkpoint):
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])

    load_checkpoint(torch.load(path))

    y_pred = []
    y_true = []
    y_prob = torch.double

    with torch.no_grad():
        for data, labels in loader:
            scores = model(data.squeeze(1))
            prob = nn.Softmax(dim=1)
            y_prob = prob(scores)
            _, prediction = scores.max(1)
            y_pred.extend(prediction)
            y_true.extend(labels)
    return y_true, y_pred, y_prob
